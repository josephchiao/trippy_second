"""Live single-pendulum display: main.py's animation, plus velocity and a movable centre.

Two things this adds over main.SP_run:

1. Velocity is shown in both places.  On the animation the cart carries a cyan
   velocity arrow next to its green force arrow, and the rolling panel below
   plots cart velocity on its own strip as it happens.  All three strips
   (position, velocity, force) roll: they hold the last WINDOW seconds and
   advance, rather than growing a static trace over the whole run.

2. The centre - the cart position the controller is trying to hold - is
   interactive.  Drag the slider or click anywhere on the track and the target
   moves there.

   The important part is *how* it moves.  The view does not rescale and the
   world does not shift: the axes stay the same 25-unit window on the same
   metres, and the cart really does travel to the new centre.  What actually
   changes is one number on the way into the network.  RL_trainer.normalize
   squashes cart position through 10 / (1 + exp(-0.4 x)) - 5, a sigmoid centred
   on x = 0, so the net has no notion of "target" at all - it only ever sees
   how far it is from wherever that sigmoid is centred, and it drives that to
   zero.  Feeding it ``x - centre`` therefore moves the point it settles at
   without touching the squash's shape, its slope, or its saturation limits.
   The net sees the same input distribution it trained on, just measured from a
   new origin.  Nothing about the network is rescaled or retrained.

   Analog control gets the same treatment through the mechanism it does have:
   the centre is written into the controller's target and its position PID.

Because the centre has to react while the sim runs, this steps the physics
inside the render loop instead of precomputing a solution array the way main.py
does.

Keeping it real time
--------------------
The physics is nearly free - about 0.1 ms per step, so a 60 Hz simulation needs
roughly 6 ms of every second.  Redrawing the figure is what costs: matplotlib
spends tens of milliseconds per full draw laying out tick labels and text, and
that is per-artist overhead, so shrinking the window or dropping the dpi buys
almost nothing.  Naively asking for 60 fps and getting 15 is what makes the
whole thing crawl in slow motion.  Three things fix it:

  * The simulation is driven by the wall clock, not by the frame count.  Each
    frame advances the physics by however much real time has passed, so a slow
    machine loses frames instead of losing speed - the pendulum still runs in
    real time, just less smoothly.  ``speed`` multiplies that rate, and the
    slider changes it live.

  * Everything that moves is blitted.  The static furniture (axes, ticks, grid,
    legends) is cached as a background and only the moving artists are redrawn,
    which costs a few milliseconds instead of tens.

  * The limits move in steps rather than continuously.  A view that pans one
    pixel per frame, or a y-axis that rescales every frame, forces a full
    redraw every frame and throws the cached background away.  So the camera
    jumps by PAN_STEP when the cart reaches the margin, the time window
    advances by SCROLL_STEP when the trace reaches the right edge, and the y
    axes rescale only when the data actually leaves them.

When the window closes, the whole run is replayed as a static figure - the
rolling strips only ever hold the last WINDOW seconds, and the shape of a
recovery is easier to judge against everything that came before it.  The centre
is drawn alongside the position, so every time you moved it shows up as a step
the cart then chases.

Run with:  python display.py
"""

import time
from collections import deque

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

import physics
import controller
import neural_network as nn
from plot_utils import timestamp_figure


refresh_rate = 60          # physics steps per simulated second, as in main.py
RENDER_FPS = 60            # frames the loop asks for; it takes what it gets
WINDOW = 10.0              # seconds held by the rolling strips
SCROLL_STEP = WINDOW / 4   # time window advances in jumps this big
CENTRE_RANGE = 10.0        # slider span for the centre, in metres
VIEW_WIDTH = 25.0          # x-span of the animation view; pans, never zooms
VIEW_MARGIN = 4.0          # pan once the cart gets this close to an edge
PAN_STEP = 8.0             # metres the view jumps when it does pan
MAX_CATCHUP = 0.25         # seconds of sim a single frame may catch up on

# Both arrows are drawn on the same budget - 4 metres of arrow at full scale,
# 100 N and 8 m/s respectively - so their lengths are comparable at a glance.
# Small values therefore draw short, which is honest but hard to read, so the
# magnitudes are also printed at the cart.
FORCE_SCALE = 0.04         # metres of arrow per newton
VEL_SCALE = 0.5            # metres of arrow per m/s
MAX_FORCE = 100.0


class MLPolicy:
    """The controller's ML branch, with the network loaded once and a centre.

    controller.SP_Controller.solve_step_inverted_rod rebuilds the network from
    disk on every call, which is fine for a batch run and far too slow inside a
    live render loop.  Same arithmetic, same checkpoint, hoisted out of the
    loop - plus the centre offset, which is the whole point of this file.
    """

    def __init__(self, network = 0, max_motor_force = MAX_FORCE):
        self.NN = nn.NeuralNetwork((4, 16, 16, 1), [nn.ELU, nn.ELU, nn.sigmoid], 'mu_nn_library')
        self.NN.theta_recover(network)
        self.max_motor_force = max_motor_force

    @staticmethod
    def normalize(state, centre = 0.0):
        """RL_trainer.normalize, with cart position measured from the centre.

        Only the first term changes, and only by where it is measured from: the
        sigmoid keeps its 0.4 steepness and its +/-5 saturation, so a state one
        metre right of the centre normalizes identically no matter where the
        centre sits.
        """
        return np.array([10 / (1 + np.exp(-0.4 * (state[0] - centre))) - 5,
                         state[1] - np.pi,
                         state[2],
                         state[3]])

    def __call__(self, state, centre = 0.0):
        mu = self.NN.feedforward(self.normalize(state, centre))[-1][0][0]
        force = (mu - 0.5) * 2 * self.max_motor_force
        return float(np.clip(force, -self.max_motor_force, self.max_motor_force)), 'ML control'


class Display:

    def __init__(self, control_type = 'ML', network = 0, y0 = None, centre = 0.0,
                 speed = 1.0):
        self.control_type = control_type
        self.centre = float(centre)
        self.speed = float(speed)           # 1.0 = real time
        self.paused = False
        self.t = 0.0
        self.total_cost = 0.0
        self.state_string = ''

        y0 = [0, np.pi, 0, 0] if y0 is None else list(y0)
        self.pendulum = physics.SinglePendulum(params = (9.81, 1, 1, 1), y0 = y0,
                                               refresh_rate = refresh_rate)
        self.motor_controller = controller.SP_Controller(
            self.pendulum, target = self.centre, max_motor_force = MAX_FORCE,
            network = network)
        self.policy = MLPolicy(network) if control_type == 'ML' else None

        # Fail here rather than three frames into the animation.  The analog
        # path is broken upstream, not by this file: SP_Controller.
        # solve_step_inverted_rod reads pendulum.state[4] and [5], which is the
        # double-pendulum state layout - a SinglePendulum state is only
        # [x, th, x_dot, th_dot], so it raises IndexError on the first call.
        if control_type == 'inverted_rod':
            raise NotImplementedError(
                'analog inverted_rod is broken in controller.SP_Controller.'
                'solve_step_inverted_rod (it indexes state[4]/state[5], which '
                'only exist on the double pendulum).  Use ML or None.')
        if control_type not in ('None', 'ML'):
            raise ValueError("Invalid control type. Choose from 'None' or 'ML'.")

        # A little longer than the visible window, so a window that has just
        # jumped forward still has history to show at its left edge.
        n = int(WINDOW * refresh_rate * 1.35)
        self.hist = {k: deque(maxlen = n) for k in ('t', 'x', 'v', 'f', 'centre')}
        # The rolling strips forget; this keeps the whole run for the summary
        # figure at the end.  Five floats per step is about 2 MB an hour.
        self.full = {k: [] for k in self.hist}

        self._bg = None            # cached static background for blitting
        self._wall = None          # wall-clock time of the previous frame
        self._backlog = 0.0        # sim steps owed, carried between frames
        self._fps = float(RENDER_FPS)
        self._x_right = WINDOW     # right edge of the strips' time window
        self._closed = False       # set when the window goes away

        self._build_figure()
        self.apply_centre(self.centre)

    # region ----------------------------------------------------------- figure
    def _build_figure(self):
        self.fig = plt.figure(figsize = (13, 9))
        self.fig.canvas.manager.set_window_title('single pendulum - live display')
        timestamp_figure(self.fig)

        # Animation on top; the three rolling strips share a time axis below it.
        # The rect is sized to the 25 x 4.4 metre view so equal aspect fills it
        # instead of shrinking the axes and leaving dead margin either side.
        self.ax = self.fig.add_axes([0.05, 0.720, 0.92, 0.215])
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-VIEW_WIDTH / 2 + 2.5, VIEW_WIDTH / 2 + 2.5)
        self.ax.set_ylim(-1.7, 2.7)
        self.ax.plot([-1000, 1000], [0, 0], 'k-', lw = 2)

        self.cart_marker, = self.ax.plot([], [], 'ks', markersize = 10)
        self.rod1, = self.ax.plot([], [], 'b-', lw = 2.5)

        self.force_arrow = self.ax.quiver([0], [0], [0], [0], color = 'green',
                                          pivot = 'tail', angles = 'xy',
                                          scale_units = 'xy', scale = 1,
                                          width = 0.006, zorder = 5)
        # Velocity gets its own arrow, above the rod tip and in cyan: offset so
        # it never lies on the force arrow (they point opposite ways often
        # enough to matter), and off blue so it is never mistaken for the rod.
        self.vel_arrow = self.ax.quiver([0], [0], [0], [0], color = 'tab:cyan',
                                        pivot = 'tail', angles = 'xy',
                                        scale_units = 'xy', scale = 1,
                                        width = 0.006, zorder = 5)
        self.vel_txt = self.ax.text(0, 0, '', color = 'tab:cyan', fontsize = 8,
                                    ha = 'center', va = 'bottom', zorder = 6)
        self.force_txt = self.ax.text(0, 0, '', color = 'green', fontsize = 8,
                                      ha = 'center', va = 'top', zorder = 6)

        # Where the controller is aiming, drawn in world coordinates.
        self.centre_line = self.ax.axvline(self.centre, color = 'tab:purple',
                                           lw = 1.2, ls = '--', alpha = 0.8, zorder = 2)
        self.centre_mark, = self.ax.plot([self.centre], [0], marker = '^',
                                         markersize = 11, color = 'tab:purple',
                                         zorder = 4)
        self.centre_txt = self.ax.text(self.centre, -1.15, '', ha = 'center',
                                       color = 'tab:purple', fontsize = 8)

        # Two status lines above the animation rather than a column inside it:
        # the axes is only a few centimetres tall now, and text in there would
        # sit on the rod.
        self.status_top = self.fig.text(0.05, 0.968, '', fontsize = 9,
                                        family = 'monospace')
        self.status_bot = self.fig.text(0.05, 0.941, '', fontsize = 9,
                                        family = 'monospace')
        self.fig.text(0.97, 0.968, 'click the track to move the centre',
                      ha = 'right', fontsize = 9, color = 'tab:purple')

        self.ax_pos = self.fig.add_axes([0.07, 0.505, 0.90, 0.155])
        self.ax_vel = self.fig.add_axes([0.07, 0.340, 0.90, 0.155], sharex = self.ax_pos)
        self.ax_frc = self.fig.add_axes([0.07, 0.175, 0.90, 0.155], sharex = self.ax_pos)
        self.ax_pos.set_xlim(0, WINDOW)

        self.line_x, = self.ax_pos.plot([], [], color = 'tab:blue', lw = 1.4,
                                        label = 'cart position (m)')
        self.line_c, = self.ax_pos.plot([], [], color = 'tab:purple', lw = 1.2,
                                        ls = '--', label = 'centre (m)')
        self.ax_pos.set_ylabel('position (m)')
        self.ax_pos.set_ylim(-2, 2)

        self.line_v, = self.ax_vel.plot([], [], color = 'tab:blue', lw = 1.4,
                                        label = 'cart velocity (m/s)')
        self.ax_vel.axhline(0, color = 'k', lw = 0.5, ls = ':')
        self.ax_vel.set_ylabel('velocity (m/s)')
        self.ax_vel.set_ylim(-2, 2)

        self.line_f, = self.ax_frc.plot([], [], color = 'green', lw = 1.2,
                                        label = 'motor force (N)')
        self.ax_frc.axhline(0, color = 'k', lw = 0.5, ls = ':')
        self.ax_frc.set_ylabel('force (N)')
        self.ax_frc.set_xlabel('time (s)')
        # A trained policy often holds with well under a newton; pinned to the
        # full +/-100 N those corrections would be a flat line at zero.
        self.ax_frc.set_ylim(-5, 5)

        for ax in (self.ax_pos, self.ax_vel, self.ax_frc):
            ax.grid(True, alpha = 0.3)
            # Lower left: the centre trace usually rides along the top of the
            # position strip.
            ax.legend(loc = 'lower left', fontsize = 8)
        for ax in (self.ax_pos, self.ax_vel):
            ax.tick_params(labelbottom = False)

        # Everything that changes every frame is blitted: excluded from the
        # cached background and redrawn by hand on top of it.
        self.moving = [self.cart_marker, self.rod1, self.force_arrow,
                       self.vel_arrow, self.vel_txt, self.force_txt,
                       self.centre_line, self.centre_mark, self.centre_txt,
                       self.line_x, self.line_c, self.line_v, self.line_f,
                       self.status_top, self.status_bot]
        for artist in self.moving:
            artist.set_animated(True)

        self._build_controls()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        # Any full draw - ours, a widget's, a resize - refreshes the cache.
        self.fig.canvas.mpl_connect('draw_event', self._on_draw)
        # A queued timer callback can outlive the window; blitting onto a
        # torn-down canvas is how that shows up, so latch the close.
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def _build_controls(self):
        ax_c = self.fig.add_axes([0.20, 0.078, 0.45, 0.026])
        self.slider = Slider(ax_c, 'centre (m)', -CENTRE_RANGE, CENTRE_RANGE,
                             valinit = self.centre, color = 'tab:purple')
        self.slider.on_changed(self.on_slider)

        ax_s = self.fig.add_axes([0.20, 0.032, 0.45, 0.026])
        self.speed_slider = Slider(ax_s, 'sim speed (x)', 0.1, 4.0,
                                   valinit = self.speed, color = 'tab:gray')
        self.speed_slider.on_changed(self.on_speed)

        ax_zero = self.fig.add_axes([0.695, 0.062, 0.07, 0.04])
        self.btn_zero = Button(ax_zero, 'centre 0')
        self.btn_zero.on_clicked(lambda _e: self.slider.set_val(0.0))

        ax_here = self.fig.add_axes([0.775, 0.062, 0.08, 0.04])
        self.btn_here = Button(ax_here, 'centre = cart')
        self.btn_here.label.set_fontsize(8)
        self.btn_here.on_clicked(
            lambda _e: self.slider.set_val(np.clip(self.pendulum.state[0],
                                                   -CENTRE_RANGE, CENTRE_RANGE)))

        ax_pause = self.fig.add_axes([0.865, 0.062, 0.07, 0.04])
        self.btn_pause = Button(ax_pause, 'pause')
        self.btn_pause.on_clicked(self.on_pause)
    # endregion

    # region ------------------------------------------------------------ centre
    def apply_centre(self, value):
        """Point everything that consumes a target at the new centre.

        The slider is clamped, but a click is not: clicking far down the rail is
        a legitimate way to send the cart somewhere the slider cannot reach, and
        the offset is just as valid out there - the sigmoid saturates the same
        way at any origin.
        """
        self.centre = float(value)
        self.motor_controller.target = self.centre
        self.motor_controller.position_controller.target = self.centre
        self.motor_controller.position_controller_1.target = self.centre
        self.centre_line.set_xdata([self.centre, self.centre])
        self.centre_mark.set_data([self.centre], [0])
        self.centre_txt.set_x(self.centre)
        self.centre_txt.set_text(f'centre {self.centre:+.2f} m')

    def on_slider(self, value):
        self.apply_centre(value)

    def on_speed(self, value):
        self.speed = float(value)

    def on_click(self, event):
        """Click on the animation to drop the centre where the pointer is."""
        if event.inaxes is not self.ax or event.xdata is None:
            return
        self.apply_centre(event.xdata)
        # Keep the slider showing the truth where it can; outside its span it
        # parks at the end and the readouts carry the real value.
        self.slider.eventson = False
        self.slider.set_val(float(np.clip(event.xdata, -CENTRE_RANGE, CENTRE_RANGE)))
        self.slider.eventson = True

    def on_pause(self, _event):
        self.paused = not self.paused
        self.btn_pause.label.set_text('run' if self.paused else 'pause')
        self._wall = None      # do not bank the paused seconds as backlog
    # endregion

    # region -------------------------------------------------------------- sim
    def step(self):
        """One physics step under the current control law and centre."""
        if self.control_type == 'None':
            force, cost = 0.0, 0.0
            self.state_string = 'None'
        else:
            force, self.state_string = self.policy(self.pendulum.state, self.centre)
            cost = (self.centre - self.pendulum.state[0]) ** 2

        self.pendulum.motor_force = force
        self.pendulum.rk4_step()
        self.t += 1 / refresh_rate
        self.total_cost += cost

        state = self.pendulum.state
        for key, value in (('t', self.t), ('x', state[0]), ('v', state[2]),
                           ('f', force), ('centre', self.centre)):
            self.hist[key].append(value)
            self.full[key].append(value)

    def advance(self):
        """Step the physics by however much real time has passed.

        Frame count is not a clock: if a draw takes 60 ms, stepping once per
        frame runs the pendulum at a quarter speed.  Tracking the wall clock
        instead means a slow machine drops smoothness rather than speed.  The
        backlog carries the fractional remainder so no time is lost to rounding,
        and MAX_CATCHUP caps a single frame so a long stall - dragging the
        window, a breakpoint - cannot trigger a burst of thousands of steps.
        """
        now = time.perf_counter()
        if self._wall is None:              # first frame, or resuming
            self._wall = now
            return
        elapsed = now - self._wall
        self._wall = now
        if self.paused:
            return

        self._fps += 0.1 * (1.0 / max(elapsed, 1e-6) - self._fps)
        self._backlog += min(elapsed, MAX_CATCHUP) * refresh_rate * self.speed
        steps = int(self._backlog)
        self._backlog -= steps
        for _ in range(steps):
            self.step()
    # endregion

    # region -------------------------------------------------------- rendering
    def _on_close(self, _event):
        self._closed = True

    def _on_draw(self, _event):
        """Cache the freshly drawn static furniture for the blit path."""
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _update_animation(self):
        """Move the cart, arrows and readouts.  True if the view had to pan."""
        state = self.pendulum.state
        x_c, th1, v = state[0], state[1], state[2]
        f = self.pendulum.motor_force

        # Pan, never zoom: the window keeps its width so distances on screen
        # keep meaning the same thing while the centre moves around.  It jumps
        # by PAN_STEP rather than tracking the cart continuously - a view that
        # slides every frame invalidates the blit background every frame.
        panned = False
        xmin, xmax = self.ax.get_xlim()
        while x_c > xmax - VIEW_MARGIN:
            xmin, xmax = xmin + PAN_STEP, xmax + PAN_STEP
            panned = True
        while x_c < xmin + VIEW_MARGIN:
            xmin, xmax = xmin - PAN_STEP, xmax - PAN_STEP
            panned = True
        if panned:
            self.ax.set_xlim(xmin, xmax)

        L = self.pendulum.params[3]
        self.cart_marker.set_data([x_c], [0])
        self.rod1.set_data([x_c, x_c + L * np.sin(th1)], [0, -L * np.cos(th1)])

        self.force_arrow.set_offsets([[x_c, 0]])
        self.force_arrow.set_UVC(f * FORCE_SCALE, 0)
        self.vel_arrow.set_offsets([[x_c, 1.6]])
        self.vel_arrow.set_UVC(v * VEL_SCALE, 0)
        self.vel_txt.set_position((x_c + v * VEL_SCALE / 2, 1.72))
        self.vel_txt.set_text(f'v {v:+.2f} m/s')
        self.force_txt.set_position((x_c + f * FORCE_SCALE / 2, -0.22))
        self.force_txt.set_text(f'F {f:+.1f} N')

        self.status_top.set_text(
            f'time {self.t:7.2f} s    location {x_c:+7.2f} m    '
            f'angle {th1:+8.4f} rad    force {f:+7.2f} N    '
            f'state: {self.state_string}')
        self.status_bot.set_text(
            f'velocity {v:+7.2f} m/s    '
            f'centre {self.centre:+7.2f} m    error {x_c - self.centre:+7.2f} m    '
            f'net input x = {10 / (1 + np.exp(-0.4 * (x_c - self.centre))) - 5:+6.3f}'
            f'    [{self._fps:4.1f} fps, sim x{self.speed:.2f}'
            f'{", paused" if self.paused else ""}]')
        return panned

    def _fit(self, ax, lo, hi, floor):
        """Rescale only when the data has actually left the axis.

        Refitting every frame would be a full redraw every frame: new limits
        mean new tick values, and re-laying out tick label text is most of what
        a matplotlib draw costs.  Hysteresis - grow when the data escapes,
        shrink only once it is rattling around in the bottom third - keeps the
        ticks still for seconds at a time.
        """
        cur_lo, cur_hi = ax.get_ylim()
        if lo >= cur_lo and hi <= cur_hi and (hi - lo) > 0.35 * (cur_hi - cur_lo):
            return False
        pad = 0.15 * max(hi - lo, floor)
        ax.set_ylim(lo - pad, hi + pad)
        return True

    def _update_strips(self):
        """Refresh the rolling traces.  True if any axis limit moved."""
        t = np.fromiter(self.hist['t'], float)
        if len(t) == 0:
            return False
        self.line_x.set_data(t, np.fromiter(self.hist['x'], float))
        self.line_c.set_data(t, np.fromiter(self.hist['centre'], float))
        self.line_v.set_data(t, np.fromiter(self.hist['v'], float))
        self.line_f.set_data(t, np.fromiter(self.hist['f'], float))

        dirty = False
        # The time window advances in SCROLL_STEP jumps once the trace reaches
        # the right edge, so the x tick labels hold still between jumps.
        if t[-1] > self._x_right:
            while self._x_right < t[-1]:
                self._x_right += SCROLL_STEP
            self.ax_pos.set_xlim(self._x_right - WINDOW, self._x_right)
            dirty = True

        pos = list(self.hist['x']) + list(self.hist['centre'])
        dirty |= self._fit(self.ax_pos, min(pos), max(pos), 1.0)
        dirty |= self._fit(self.ax_vel, min(self.hist['v']), max(self.hist['v']), 1.0)
        # Force is kept symmetric about zero: a push left and a push right of
        # the same size should look the same size.
        peak = max(max(abs(v) for v in self.hist['f']), 1.0)
        dirty |= self._fit(self.ax_frc, -peak, peak, 2.0)
        return dirty

    def frame(self):
        """One pass: advance the sim, then repaint as cheaply as possible."""
        if self._closed:
            return
        self.advance()
        dirty = self._update_animation()
        dirty = self._update_strips() or dirty

        canvas = self.fig.canvas
        if dirty or self._bg is None:
            canvas.draw()               # limits moved; _on_draw recaches
        else:
            canvas.restore_region(self._bg)
        for artist in self.moving:
            if artist.axes is None:
                self.fig.draw_artist(artist)
            else:
                artist.axes.draw_artist(artist)
        canvas.blit(self.fig.bbox)
        canvas.flush_events()
    # endregion

    def summary(self):
        """The whole run on one static figure, drawn after the window closes."""
        if len(self.full['t']) == 0:
            print('nothing to plot - no steps were taken')
            return
        t = np.array(self.full['t'])
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (12, 8), sharex = True,
                                            constrained_layout = True)
        fig.canvas.manager.set_window_title('single pendulum - full run')
        fig.suptitle(f'full run - control: {self.control_type}, '
                     f'{t[-1]:.1f} s simulated  (cost = {self.total_cost:.2f})')
        timestamp_figure(fig)

        centre = np.array(self.full['centre'])
        ax1.plot(t, self.full['x'], color = 'tab:blue', lw = 1.2,
                 label = 'cart position (m)')
        ax1.plot(t, centre, color = 'tab:purple', lw = 1.2, ls = '--',
                 label = 'centre (m)')
        # Mark every move of the centre, so a settling time can be read off
        # against the moment the target actually changed.
        for i in np.flatnonzero(np.diff(centre) != 0.0):
            ax1.axvline(t[i + 1], color = 'tab:purple', lw = 0.6, alpha = 0.35)
        ax1.set_ylabel('position (m)')

        ax2.plot(t, self.full['v'], color = 'tab:blue', lw = 1.0,
                 label = 'cart velocity (m/s)')
        ax2.axhline(0, color = 'k', lw = 0.5, ls = ':')
        ax2.set_ylabel('velocity (m/s)')

        ax3.plot(t, self.full['f'], color = 'green', lw = 1.0,
                 label = 'motor force (N)')
        ax3.axhline(0, color = 'k', lw = 0.5, ls = ':')
        ax3.set_ylabel('force (N)')
        ax3.set_xlabel('time (s)')

        for ax in (ax1, ax2, ax3):
            ax.grid(True, alpha = 0.3)
            ax.legend(loc = 'upper right', fontsize = 8)
        return fig

    def run(self, fps = RENDER_FPS, static_plot = True):
        # A plain canvas timer rather than FuncAnimation: with blit=False the
        # animation calls draw_idle() after every frame, which would force the
        # full redraw this whole design exists to avoid.
        self.fig.canvas.draw()
        self.timer = self.fig.canvas.new_timer(interval = int(1000 / fps))
        self.timer.add_callback(self.frame)
        self.timer.start()
        plt.show()

        # plt.show() returns once the live window is closed, so the summary is
        # built from a finished run rather than raced against one.
        self.timer.stop()
        print('cost =', self.total_cost)
        if static_plot and self.summary() is not None:
            plt.show()


def SP_display(control_type = 'ML', network = 0, y0 = None, centre = 0.0,
               speed = 1.0, static_plot = True):
    """Live version of main.SP_run with velocity readouts and a movable centre."""
    Display(control_type = control_type, network = network, y0 = y0,
            centre = centre, speed = speed).run(static_plot = static_plot)


if __name__ == '__main__':
    SP_display(control_type = 'ML', network = 0, y0 = [0, np.pi, 0, 0], centre = 0.0)
