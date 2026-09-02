"""Interactive activation viewer for the actor and critic networks.

Drag the four input sliders and watch which hidden units light up.  Both nets
use ReLU hidden layers, so a unit is literally a switch: it is *off* whenever
its pre-activation ``z = a_prev . w + b`` is <= 0, and contributes nothing
downstream.  That makes "which nodes are on for this state" a well-defined
question, and this tool answers it live.

Reading the picture
-------------------
  * Node fill   - grey/hollow = off (z <= 0); warm colour = on, brightness
                  scaled by the activation relative to the rest of its layer.
  * Cyan ring   - the unit is near its switching point (|z| small), so a small
                  nudge of any slider will flip it on or off.
  * Edge alpha  - live contribution |a_src * w|, not just the raw weight, so
                  the visible web is the path the current state actually takes.
  * Firing-rate mode recolours every node by the fraction of randomly sampled
    states that switch it on.  Nodes that come out at 0.00 are dead ReLUs -
    they fire nowhere in the state space and are pure dead weight.

The architectures below mirror ``RL_trainer.__init__``; if you change the dims
there, change them here too.  Weights are read from the same checkpoint slots
the trainer writes, so re-running this after training shows the current net.

Angle convention: theta = pi is upright (12 o'clock) and positive theta is
counter-clockwise, so the ``th-pi`` input is positive when the rod leans left.
Positive motor force pushes the cart right.

Run with:  python network_visualizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Wedge
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.widgets import Slider, RadioButtons, Button

import neural_network as nn
from plot_utils import timestamp_figure


# --- kept in sync with RL_training.RL_trainer.__init__ ------------------------
NETWORKS = {
    'actor  (mu)': dict(dim=(4, 16, 16, 1),
                        norm_fcn=[nn.ELU, nn.ELU, nn.sigmoid],
                        location='mu_nn_library'),
    'critic  (V)': dict(dim=(4, 64, 64, 1),
                        norm_fcn=[nn.ELU, nn.ELU, nn.linear],
                        location='V_nn_library'),
}

# RL_training.normalize() feeds the net [x, theta - pi, x_dot, theta_dot].
#
# Angle convention (see physics.SinglePendulum: x1 = x + L sin th, y1 = -L cos th):
# theta = 0 hangs straight down and theta = pi is upright at 12 o'clock, with
# positive theta running counter-clockwise.  So a positive theta - pi means the
# rod has tipped counter-clockwise, which on screen is to the LEFT.  Positive
# cart position, cart velocity and motor force are all to the RIGHT.
#
# The +/- pi/2 angle range is exactly the band the trainer treats as alive - it
# calls the episode done at theta <= pi/2 or theta >= 3*pi/2 - so any angle you
# can dial in here is one the net is actually asked to handle.
#
# Entries are (short name for the node, slider caption, min, max).
INPUTS = [
    ('x',     'cart position (m)    + right',     -10.0,       10.0),
    ('th-pi', 'tilt from upright    + CCW/left',  -np.pi / 2,  np.pi / 2),
    ('xdot',  'cart velocity (m/s)  + right',     -8.0,        8.0),
    ('thdot', 'angular velocity     + CCW/left',  -8.0,        8.0),
]

MAX_EDGES = 600    # strongest-|w| connections drawn; the full 4416 is visual mush
SAMPLES = 4000     # random states used to estimate per-node firing rates
NEAR_EDGE = 0.08   # |z| below this fraction of the layer's z-scale = "about to flip"
ACTOR_FORCE = 100.0  # RL_training maps mu in [0,1] to (mu - 0.5) * 200 newtons
OUT_GAMMA = 0.6    # colour-ramp gamma for signed nodes; see signed_warp()
DIAL_INPUT = 1     # index into INPUTS driven by the angle dial rather than a slider

OFF_FACE = '#2a2f3a'
OFF_EDGE = '#555c6b'
ON_CMAP = plt.get_cmap('inferno')
RATE_CMAP = plt.get_cmap('viridis')
# Signed quantities (inputs, and the output's force/value) run blue-dark-red.
# Zero maps to a dark neutral rather than white, so on this dark canvas "no
# force" reads as an unlit node, the same visual language as an off ReLU.
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    'signed', ['#4a9df5', '#39404e', '#ff7043'])
# The critic's value head is not a signed quantity with a natural symmetric
# span, so it gets a plain low-to-high sequential ramp instead: dark = a state
# the critic rates poorly, bright = one it rates well.
VALUE_CMAP = LinearSegmentedColormap.from_list(
    'value', ['#1b2130', '#4b2a6b', '#a03f8c', '#f07fae', '#ffd7c2'])


def apply_activation(fn, z):
    """Run a layer's activation, handling the per-column list form the nets allow."""
    if isinstance(fn, list):
        return np.column_stack([fn[j](z[:, j]) for j in range(len(fn))])
    return fn(z)


class AngleDial:
    """Circular stand-in for the tilt slider: the rod points where you drag it.

    A linear slider makes you translate a number into a pose in your head.  Here
    the widget *is* the pose - the drawn rod sits at the angle the net is being
    fed, on the same geometry as physics.SinglePendulum (theta = pi straight up,
    positive theta counter-clockwise, so positive theta - pi leans left).

    Exposes the slice of the matplotlib Slider API the rest of the app uses
    (val / set_val / reset / on_changed) so it drops in where a Slider was.
    """

    def __init__(self, ax, vmin, vmax, valinit=0.0, caption=''):
        self.ax = ax
        self.vmin, self.vmax = float(vmin), float(vmax)
        self.valinit = float(valinit)
        self.val = float(valinit)
        self._cbs = []
        self._dragging = False
        self._draw_static(caption)
        self._redraw()
        canvas = ax.figure.canvas
        canvas.mpl_connect('button_press_event', self._on_press)
        canvas.mpl_connect('motion_notify_event', self._on_motion)
        canvas.mpl_connect('button_release_event', self._on_release)

    # -- Slider-compatible surface ---------------------------------------------
    def on_changed(self, fn):
        self._cbs.append(fn)

    def set_val(self, v):
        self.val = float(np.clip(v, self.vmin, self.vmax))
        self._redraw()
        for fn in self._cbs:
            fn(self.val)

    def reset(self):
        self.set_val(self.valinit)

    # -- geometry ---------------------------------------------------------------
    @staticmethod
    def tip(d):
        """Rod tip for a tilt d, matching x1 = x + L sin(th), y1 = -L cos(th).

        With th = pi + d that is (-sin d, cos d): d = 0 points straight up and
        increasing d swings the tip left, i.e. counter-clockwise.
        """
        return -np.sin(d), np.cos(d)

    def _draw_static(self, caption):
        ax = self.ax
        ax.set_aspect('equal')
        ax.set_xlim(-1.38, 1.38)
        ax.set_ylim(-1.38, 1.38)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#1b1f27')
        for sp in ax.spines.values():
            sp.set_visible(False)

        # Below the horizon the trainer has already called the episode done, so
        # those angles are unreachable here - shade them rather than hide them,
        # to show the dial is a window onto the live band, not the whole circle.
        ax.add_patch(Wedge((0, 0), 1.0, 180, 360, color='#e06c6c',
                           alpha=0.07, zorder=1))
        full = np.linspace(0, 2 * np.pi, 240)
        ax.plot(np.cos(full), np.sin(full), color='#2a2f3a', lw=1, zorder=1)
        band = np.linspace(self.vmin, self.vmax, 160)
        bx, by = self.tip(band)
        ax.plot(bx, by, color='#4c5769', lw=3, solid_capstyle='round', zorder=2)
        ax.plot([-1.06, 1.06], [0, 0], color='#e06c6c', lw=0.8, ls=':',
                alpha=0.7, zorder=2)
        ax.text(0, -1.16, 'fallen', ha='center', va='center',
                color='#e06c6c', fontsize=7, alpha=0.85)

        for d in (-np.pi / 2, -np.pi / 4, 0.0, np.pi / 4, np.pi / 2):
            tx, ty = self.tip(d)
            ax.plot([tx * 1.02, tx * 1.10], [ty * 1.02, ty * 1.10],
                    color='#5c6675', lw=1, zorder=2)
            upright = d == 0.0
            ax.text(tx * 1.24, ty * 1.24,
                    'upright' if upright else f'{np.degrees(d):+.0f}',
                    ha='center', va='center', fontsize=7,
                    color='#7fd48c' if upright else '#8d97a8')

        # Cart at the pivot, so left/right on the dial reads as left/right on screen.
        ax.add_patch(Rectangle((-0.17, -0.10), 0.34, 0.20, color='#39414f', zorder=4))
        self.rod, = ax.plot([0, 0], [0, 1], color='#f0d68c', lw=3,
                            solid_capstyle='round', zorder=5)
        self.handle, = ax.plot([0], [1], marker='o', markersize=9,
                               color='#ffd7a0', markeredgecolor='#1b1f27', zorder=6)

        # Caption and readout live outside the axes so they cost the dial no radius.
        box = ax.get_position()
        mid = box.x0 + box.width / 2
        ax.figure.text(mid, box.y1 + 0.008, caption, ha='center', va='bottom',
                       color='#8d97a8', fontsize=8)
        self.txt = ax.figure.text(mid, box.y0 - 0.010, '', ha='center', va='top',
                                  color='#d8dde6', fontsize=8, family='monospace')

    def _redraw(self):
        x, y = self.tip(self.val)
        self.rod.set_data([0, x], [0, y])
        self.handle.set_data([x], [y])
        self.txt.set_text(f'{self.val:+.3f} rad  ({np.degrees(self.val):+5.1f} deg)\n'
                          f'theta = {np.pi + self.val:.3f}')

    # -- interaction -------------------------------------------------------------
    def _coords(self, event):
        """Data coords for the event, tracking the pointer outside the axes too."""
        if event.inaxes is self.ax and event.xdata is not None:
            return event.xdata, event.ydata
        if event.x is None:
            return None
        return tuple(self.ax.transData.inverted().transform((event.x, event.y)))

    def _apply(self, event):
        xy = self._coords(event)
        if xy is None:
            return
        mx, my = xy
        if np.hypot(mx, my) < 0.12:      # too close to the pivot to mean an angle
            return
        # Inverse of tip(): the drag direction read back as a tilt, then clamped
        # to the live band so dragging below the horizon parks it on the boundary.
        self.set_val(np.arctan2(-mx, my))

    def _on_press(self, event):
        if event.inaxes is self.ax:
            self._dragging = True
            self._apply(event)

    def _on_motion(self, event):
        if self._dragging:
            self._apply(event)

    def _on_release(self, _event):
        self._dragging = False


def signed_warp(p, gamma=OUT_GAMMA):
    """Bend a [0,1] signed position toward the ends of the colour ramp.

    Only the *colour* is warped, never the number: the scale bar stays a linear
    force axis and its ticks stay exact, because 0, 0.5 and 1 are fixed points.
    Without this a policy commanding +/-10 N of its +/-100 N authority would sit
    on the ramp's neutral centre and look identical to zero force.
    """
    d = 2 * np.asarray(p, dtype=float) - 1
    return 0.5 + 0.5 * np.sign(d) * np.abs(d) ** gamma


def gate_kind(fn):
    """How a layer's activation switches, which decides what 'off' means."""
    name = getattr(fn, '__name__', '')
    if name in ('ReLU', 'ELU', 'LeakyReLU'):
        return 'gated'        # hard (or near-hard) off at z <= 0
    if name == 'sigmoid':
        return 'saturating'   # no true off; call a < 0.5 the low state
    return 'linear'           # ungated, always passing signal


def forward_trace(theta, b, norm_fcn, X):
    """Forward pass that keeps the pre-activations, which feedforward() drops.

    X is (N, n_in).  Returns (zs, activations) where zs[i] is the (N, n) matrix
    of pre-activations entering layer i+1 and activations[0] is X itself.
    """
    a = np.atleast_2d(np.asarray(X, dtype=float))
    zs, acts = [], [a]
    for i in range(len(theta)):
        z = np.dot(a, theta[i]) + b[i]
        a = apply_activation(norm_fcn[i], z)
        zs.append(z)
        acts.append(a)
    return zs, acts


class NetworkVisualizer:

    def __init__(self, net_name='actor  (mu)', slot=0):
        self.net_name = net_name
        self.slot = slot
        self.color_mode = 'activation'
        self.values = np.array([0.0, 0.0, 0.0, 0.0])

        self.fig = plt.figure(figsize=(14, 9.0))
        self.fig.canvas.manager.set_window_title('NN activation viewer')
        timestamp_figure(self.fig, color='0.55')
        self.ax = self.fig.add_axes([0.04, 0.32, 0.80, 0.64])
        self.ax.set_facecolor('#12151c')
        self.fig.patch.set_facecolor('#1b1f27')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        self._build_controls()
        self.load_network()
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)

    # region ---------------------------------------------------------- controls
    def _build_controls(self):
        # self.sliders stays indexed by INPUTS position; the tilt entry holds an
        # AngleDial rather than a Slider, but presents the same interface.
        self.sliders = [None] * len(INPUTS)
        self.zero_buttons = [None] * len(INPUTS)
        row = 0
        for k, (short, caption, lo, hi) in enumerate(INPUTS):
            if k == DIAL_INPUT:
                ax_d = self.fig.add_axes([0.035, 0.075, 0.19, 0.205])
                widget = AngleDial(ax_d, lo, hi, 0.0, caption=f'{short}   {caption}')
                ax_z = self.fig.add_axes([0.100, 0.010, 0.06, 0.028])
                zero_label = 'upright'
            else:
                y = 0.185 - 0.055 * row
                ax_s = self.fig.add_axes([0.355, y, 0.175, 0.028])
                ax_s.set_facecolor('#2a2f3a')
                widget = Slider(ax_s, f'{short}   {caption}', lo, hi,
                                valinit=0.0, color='#4c8fbd')
                widget.label.set_fontsize(8)
                widget.label.set_color('#d8dde6')
                widget.valtext.set_color('#d8dde6')
                ax_z = self.fig.add_axes([0.585, y, 0.03, 0.028])
                zero_label = '0'
                row += 1

            widget.on_changed(self.on_slider)
            self.sliders[k] = widget

            # Per-input zero button, so one input can be neutralised without
            # losing the rest of the state you dialled in.
            btn = Button(ax_z, zero_label, color='#39414f', hovercolor='#4c5769')
            btn.label.set_color('#d8dde6')
            btn.label.set_fontsize(7 if zero_label != '0' else 8)
            btn.on_clicked(lambda _evt, w=widget: w.reset())
            self.zero_buttons[k] = btn   # keep a reference or it stops firing

        def panel(rect, labels, active, cb):
            ax_r = self.fig.add_axes(rect, facecolor='#2a2f3a')
            r = RadioButtons(ax_r, labels, active=active)
            for t in r.labels:
                t.set_color('#d8dde6')
                t.set_fontsize(8)
            r.on_clicked(cb)
            return r

        self.radio_net = panel([0.62, 0.145, 0.15, 0.085],
                               list(NETWORKS), list(NETWORKS).index(self.net_name),
                               self.on_network)
        self.radio_slot = panel([0.62, 0.025, 0.15, 0.10],
                                ['slot 0  (best)', 'slot 1  (prev)', 'slot 2  (periodic)'],
                                self.slot, self.on_slot)
        self.radio_mode = panel([0.80, 0.025, 0.17, 0.11],
                                ['activation', 'firing rate'], 0, self.on_mode)

        ax_rand = self.fig.add_axes([0.80, 0.185, 0.08, 0.04])
        self.btn_rand = Button(ax_rand, 'random', color='#39414f', hovercolor='#4c5769')
        self.btn_rand.label.set_color('#d8dde6')
        self.btn_rand.on_clicked(self.on_random)

        ax_reset = self.fig.add_axes([0.89, 0.185, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'reset', color='#39414f', hovercolor='#4c5769')
        self.btn_reset.label.set_color('#d8dde6')
        self.btn_reset.on_clicked(self.on_reset)
    # endregion

    # region ------------------------------------------------------ net loading
    def load_network(self):
        """Pull weights off disk and rebuild the whole diagram."""
        cfg = NETWORKS[self.net_name]
        net = nn.NeuralNetwork(cfg['dim'], cfg['norm_fcn'], cfg['location'])
        try:
            net.theta_recover(self.slot)
        except FileNotFoundError:
            self.ax.clear()
            self.ax.set_axis_off()
            self.ax.text(0.5, 0.5, f"no checkpoint in {cfg['location']} slot {self.slot}",
                         ha='center', color='#e06c6c', transform=self.ax.transAxes)
            self.fig.canvas.draw_idle()
            return

        self.dim = cfg['dim']
        self.norm_fcn = cfg['norm_fcn']
        self.theta = net.theta
        self.b = net.b
        self.kinds = [gate_kind(f) for f in self.norm_fcn]
        self.is_actor = self.net_name.startswith('actor')

        self._layout_nodes()
        self._select_edges()
        self._sample_firing_rates()
        self._draw_static()
        self.update()

    def _layout_nodes(self):
        """One column per layer, nodes spread over a fixed vertical span."""
        self.pos = []
        for i, n in enumerate(self.dim):
            y = np.zeros(1) if n == 1 else np.linspace(1.0, -1.0, n)
            self.pos.append(np.column_stack([np.full(n, float(i)), y]))
        self.node_size = float(np.clip(2600.0 / max(self.dim), 10.0, 260.0))

    def _select_edges(self):
        """Keep the strongest |w| connections; drawing all of them hides the signal."""
        entries = []
        for i, w in enumerate(self.theta):
            src, dst = np.meshgrid(np.arange(w.shape[0]), np.arange(w.shape[1]), indexing='ij')
            entries.append(np.column_stack([
                np.full(w.size, i), src.ravel(), dst.ravel(), w.ravel()]))
        allw = np.vstack(entries)
        if len(allw) > MAX_EDGES:
            keep = np.argsort(np.abs(allw[:, 3]))[-MAX_EDGES:]
            allw = allw[keep]

        self.e_layer = allw[:, 0].astype(int)
        self.e_src = allw[:, 1].astype(int)
        self.e_dst = allw[:, 2].astype(int)
        self.e_w = allw[:, 3]
        starts = np.stack([self.pos[l][s] for l, s in zip(self.e_layer, self.e_src)])
        ends = np.stack([self.pos[l + 1][d] for l, d in zip(self.e_layer, self.e_dst)])
        self.segments = np.stack([starts, ends], axis=1)

    def _sample_firing_rates(self):
        """Fraction of random states that switch each hidden unit on."""
        lows = np.array([lo for _, _, lo, _ in INPUTS])
        highs = np.array([hi for _, _, _, hi in INPUTS])
        X = np.random.uniform(lows, highs, size=(SAMPLES, len(INPUTS)))
        zs, acts = forward_trace(self.theta, self.b, self.norm_fcn, X)
        self.firing_rate = [np.ones(self.dim[0])]
        for i, z in enumerate(zs):
            if self.kinds[i] == 'gated':
                self.firing_rate.append((z > 0).mean(axis=0))
            elif self.kinds[i] == 'saturating':
                self.firing_rate.append((acts[i + 1] > 0.5).mean(axis=0))
            else:
                self.firing_rate.append(np.full(z.shape[1], np.nan))
        self._set_output_scale(acts[-1])

    def _set_output_scale(self, sampled_out):
        """Pick the output node's colour scale.  The two heads need different ones.

        The actor's sigmoid stands in for a signed motor force: zero is a real,
        physically special point and +/-100 N is a hard limit, so it gets a
        diverging ramp pinned symmetrically about zero, plus a gamma because the
        forces it actually commands bunch up near the middle of that fixed span.

        The critic's linear head has none of that structure.  Its value estimate
        has no fixed scale, no hard limit and no reason to be symmetric - here it
        runs about -220 to +50, so forcing the actor's symmetric-about-zero ramp
        onto it wastes the whole upper third of the colour range and leaves every
        well-rated state looking washed out.  It gets a sequential ramp fitted to
        the range the critic actually produces, and no gamma, since that range is
        already fitted to the data.
        """
        values = self.out_value(sampled_out)
        # p1/p99 rather than min/max so one outlier state cannot flatten the rest.
        self.out_sorted = np.sort(np.asarray(values).ravel())
        if self.is_actor:
            ref = min(max(float(np.percentile(np.abs(values), 99)), 1e-9), ACTOR_FORCE)
            self.out_label = 'motor force'
            self.out_unit = ' N'
            self.out_lo, self.out_hi = -ref, ref
            self.out_cmap = SIGNED_CMAP
            self.out_warp = signed_warp
            self.out_ticks = [(-ref, 'push left'), (0.0, ''), (ref, 'push right')]
        else:
            lo = float(np.percentile(values, 1))
            hi = float(np.percentile(values, 99))
            if hi - lo < 1e-9:
                lo, hi = lo - 0.5, hi + 0.5
            self.out_label = 'value  (critic estimate)'
            self.out_unit = ''
            self.out_lo, self.out_hi = lo, hi
            self.out_cmap = VALUE_CMAP
            self.out_warp = lambda p: np.asarray(p, dtype=float)
            # "seen" because these are the ends of the sampled state box, not
            # absolute bounds - V has no absolute bounds.
            self.out_ticks = [(lo, 'worst seen'), (hi, 'best seen')]
            if lo < 0.0 < hi:
                self.out_ticks.insert(1, (0.0, ''))

    def _frac_of(self, value):
        """Where a value in display units sits on the scale bar, linearly in 0..1."""
        return float(np.clip((value - self.out_lo) / (self.out_hi - self.out_lo), 0, 1))

    def _out_frac(self, a):
        """Where an output activation sits on the scale bar, linearly in 0..1."""
        return self._frac_of(self.out_value(a))

    def out_percentile(self, a):
        """How this output ranks against the sampled state box, in percent."""
        i = int(np.searchsorted(self.out_sorted, self.out_value(a)))
        return 100.0 * i / len(self.out_sorted)

    def out_value(self, a):
        """Raw output activation -> the signed quantity it actually represents."""
        return (a - 0.5) * 2 * ACTOR_FORCE if self.is_actor else a
    # endregion

    # region -------------------------------------------------------- rendering
    def _draw_static(self):
        self.ax.clear()
        self.ax.set_facecolor('#12151c')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.set_xlim(-0.55, len(self.dim) - 0.45)
        self.ax.set_ylim(-1.68, 1.30)

        self.edge_coll = LineCollection(self.segments, linewidths=0.8, zorder=1)
        self.ax.add_collection(self.edge_coll)

        self.scatters = []
        for i, p in enumerate(self.pos):
            sc = self.ax.scatter(p[:, 0], p[:, 1], s=self.node_size,
                                 facecolors=OFF_FACE, edgecolors=OFF_EDGE,
                                 linewidths=0.8, zorder=3)
            self.scatters.append(sc)

        names = ['input'] + [f'hidden {i + 1}' for i in range(len(self.dim) - 2)] + ['output']
        self.layer_txt = []
        for i, n in enumerate(self.dim):
            self.ax.text(i, 1.14, f"{names[i]}\n{n} node{'s' if n > 1 else ''}", ha='center', va='bottom',
                         color='#8d97a8', fontsize=8)
            self.layer_txt.append(
                self.ax.text(i, -1.16, '', ha='center', va='top',
                             color='#d8dde6', fontsize=8))

        # Input node labels, so it is obvious which slider drives which node.
        for k, (short, _, _, _) in enumerate(INPUTS):
            self.ax.text(-0.12, self.pos[0][k, 1], short,
                         ha='right', va='center', color='#9fb4c9', fontsize=9)

        self._draw_output_key()

        self.readout = self.ax.text(len(self.dim) / 2 - 0.5, -1.62, '',
                                    ha='center', va='bottom', color='#f0d68c',
                                    fontsize=10, family='monospace')

        self.tip = self.ax.annotate(
            '', xy=(0, 0), xytext=(12, 12), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.4', fc='#39414f', ec='#7d8899'),
            color='#eef2f7', fontsize=8, family='monospace', zorder=6, visible=False)

    def _draw_output_key(self):
        """Gradient strip under the output node, giving its colour a scale."""
        xo = len(self.dim) - 1
        half = 0.42
        self.out_key_span = (xo, 2 * half)
        self.ax.imshow(self.out_warp(np.linspace(0, 1, 256)).reshape(1, -1),
                       extent=[xo - half, xo + half, -1.34, -1.27],
                       aspect='auto', cmap=self.out_cmap, zorder=4)
        # Mark zero on the bar wherever it falls, which is off-centre for the
        # critic because its value range is not symmetric.
        if self.out_lo < 0.0 < self.out_hi:
            xz = xo + (self._frac_of(0.0) - 0.5) * 2 * half
            self.ax.plot([xz, xz], [-1.34, -1.27], color='#12151c', lw=0.9, zorder=5)
        for val, side in self.out_ticks:
            dx = (self._frac_of(val) - 0.5) * 2 * half
            tick = '0' if val == 0 else f'{val:+.0f}'
            self.ax.text(xo + dx, -1.37, tick + self.out_unit + (f'\n{side}' if side else ''),
                         ha='center', va='top', color='#8d97a8', fontsize=7)
        self.ax.text(xo, -1.19, self.out_label, ha='center', va='bottom',
                     color='#8d97a8', fontsize=8)
        # Tracks where the current output sits on that scale.
        self.out_marker, = self.ax.plot([xo], [-1.25], marker='v', markersize=7,
                                        color='#eef2f7', zorder=6)

    def _place_out_marker(self):
        xo, span = self.out_key_span
        frac = float(self._out_frac(self.acts[-1][0]))
        self.out_marker.set_xdata([xo + (frac - 0.5) * span])

    def update(self, _=None):
        zs, acts = forward_trace(self.theta, self.b, self.norm_fcn, self.values)
        self.zs = [z.ravel() for z in zs]
        self.acts = [a.ravel() for a in acts]

        self._color_nodes()
        self._color_edges()
        self._place_out_marker()
        self._write_readout()
        self.fig.canvas.draw_idle()

    def _node_state(self, layer):
        """(on-mask, brightness in [0,1]) for one layer of nodes."""
        a = self.acts[layer]
        if layer == 0:
            lo = np.array([l for _, _, l, _ in INPUTS])
            hi = np.array([h for _, _, _, h in INPUTS])
            return np.ones(len(a), bool), signed_warp(np.clip((a - lo) / (hi - lo), 0, 1))

        kind = self.kinds[layer - 1]
        if layer == len(self.dim) - 1:
            # Output node: a readout, not a gate.  A sigmoid is never "off", so
            # colour it by the signed force it commands instead - blue for a
            # push one way, red for the other, neutral at zero force.
            return np.ones(len(a), bool), np.atleast_1d(self.out_warp(self._out_frac(a)))
        if kind == 'gated':
            on = self.zs[layer - 1] > 0
            scale = max(float(a.max()), 1e-9)
            return on, np.clip(a / scale, 0, 1)
        if kind == 'saturating':
            return a > 0.5, np.clip(a, 0, 1)
        # Ungated hidden layer: no on/off, so colour by signed magnitude.
        scale = max(float(np.abs(a).max()), 1e-9)
        return np.ones(len(a), bool), np.clip(0.5 + 0.5 * a / scale, 0, 1)

    def _color_nodes(self):
        for layer, sc in enumerate(self.scatters):
            n = self.dim[layer]
            if self.color_mode == 'firing rate' and 0 < layer < len(self.dim) - 1:
                rate = self.firing_rate[layer]
                faces = np.where(np.isnan(rate)[:, None],
                                 np.array(SIGNED_CMAP(0.5))[None, :],
                                 RATE_CMAP(np.nan_to_num(rate)))
                edges = np.where((np.nan_to_num(rate) == 0)[:, None],
                                 np.array([[1.0, 0.30, 0.30, 1.0]]),
                                 np.array([[0.35, 0.39, 0.45, 1.0]]))
                widths = np.where(np.nan_to_num(rate) == 0, 1.8, 0.7)
            else:
                on, bright = self._node_state(layer)
                if layer == len(self.dim) - 1:
                    faces = self.out_cmap(bright)        # head's own scale
                elif layer == 0:
                    faces = SIGNED_CMAP(bright)          # signed, symmetric by construction
                else:
                    faces = ON_CMAP(0.25 + 0.7 * bright)
                faces[~on] = to_rgba(OFF_FACE)

                # Ring the units sitting right on their switching threshold.
                edges = np.tile(to_rgba(OFF_EDGE), (n, 1))
                widths = np.full(n, 0.8)
                if layer > 0 and self.kinds[layer - 1] == 'gated':
                    z = self.zs[layer - 1]
                    scale = max(float(np.abs(z).max()), 1e-9)
                    borderline = np.abs(z) < NEAR_EDGE * scale
                    edges[borderline] = (0.35, 0.95, 0.98, 1.0)
                    widths[borderline] = 1.8

            sc.set_facecolors(faces)
            sc.set_edgecolors(edges)
            sc.set_linewidths(widths)

        for layer, txt in enumerate(self.layer_txt):
            if layer == 0 or layer == len(self.dim) - 1:
                txt.set_text('')       # input and output are readouts, not gates
                continue
            if self.color_mode == 'firing rate':
                rate = self.firing_rate[layer]
                if np.isnan(rate).all():
                    txt.set_text('ungated')
                else:
                    dead = int((np.nan_to_num(rate) == 0).sum())
                    txt.set_text(f'dead: {dead}/{self.dim[layer]}')
                    txt.set_color('#e06c6c' if dead else '#7fd48c')
            else:
                on, _ = self._node_state(layer)
                live = int(on.sum())
                txt.set_text(f'on: {live}/{self.dim[layer]}')
                txt.set_color('#7fd48c' if live else '#e06c6c')

    def _color_edges(self):
        """Alpha tracks the live contribution a_src * w, so dead paths fade out."""
        src_act = np.array([self.acts[l][s] for l, s in zip(self.e_layer, self.e_src)])
        contrib = np.abs(src_act * self.e_w)
        scale = max(float(contrib.max()), 1e-9)
        alpha = 0.03 + 0.72 * np.clip(contrib / scale, 0, 1) ** 0.6

        colors = np.zeros((len(self.e_w), 4))
        pos = self.e_w >= 0
        colors[pos] = (0.95, 0.45, 0.30, 1.0)    # excitatory
        colors[~pos] = (0.35, 0.62, 0.95, 1.0)   # inhibitory
        colors[:, 3] = alpha
        self.edge_coll.set_color(colors)
        self.edge_coll.set_linewidths(0.4 + 1.5 * alpha)

    def _write_readout(self):
        out = float(self.acts[-1][0])
        if self.is_actor:
            text = f'mu = {out:.4f}      motor force = {self.out_value(out):+.2f} N'
        else:
            text = (f'V(state) = {out:+.3f}      rated above '
                    f'{self.out_percentile(out):.0f}% of sampled states')
        live = sum(int(self._node_state(l)[0].sum())
                   for l in range(1, len(self.dim) - 1))
        total = sum(self.dim[1:-1])
        text += f'      hidden units on: {live}/{total}  ({100 * live / total:.0f}%)'
        self.readout.set_text(text)
    # endregion

    # region ---------------------------------------------------------- callbacks
    def on_slider(self, _):
        self.values = np.array([s.val for s in self.sliders])
        self.update()

    def on_network(self, label):
        self.net_name = label
        self.load_network()

    def on_slot(self, label):
        self.slot = int(label.split()[1])
        self.load_network()

    def on_mode(self, label):
        self.color_mode = label
        self.update()

    def on_random(self, _):
        for s, (_, _, lo, hi) in zip(self.sliders, INPUTS):
            s.set_val(np.random.uniform(lo, hi))

    def on_reset(self, _):
        for s in self.sliders:
            s.set_val(0.0)

    def on_hover(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            if self.tip.get_visible():
                self.tip.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        layer = int(round(event.xdata))
        if not 0 <= layer < len(self.dim) or abs(event.xdata - layer) > 0.25:
            if self.tip.get_visible():
                self.tip.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        ys = self.pos[layer][:, 1]
        idx = int(np.argmin(np.abs(ys - event.ydata)))
        if abs(ys[idx] - event.ydata) > max(0.03, 2.0 / max(self.dim)):
            if self.tip.get_visible():
                self.tip.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        a = self.acts[layer][idx]
        if layer == len(self.dim) - 1:
            body = (f'output node\n'
                    f'{self.out_label} = {self.out_value(a):+.3f}{self.out_unit}\n'
                    f'raw activation = {a:+.4f}')
            if not self.is_actor:
                body += (f'\nrated above {self.out_percentile(a):.1f}% of states'
                         f'\nsampled range {self.out_lo:+.0f} .. {self.out_hi:+.0f}')
        elif layer == 0:
            body = (f'input {idx}  ({INPUTS[idx][0]})\n'
                    f'{INPUTS[idx][1]}\nvalue = {a:+.4f}')
        else:
            z = self.zs[layer - 1][idx]
            state = 'ON ' if self._node_state(layer)[0][idx] else 'OFF'
            body = (f'layer {layer}  node {idx}\n'
                    f'z = {z:+.4f}   [{state}]\n'
                    f'a = {a:+.4f}')
            rate = self.firing_rate[layer][idx]
            if not np.isnan(rate):
                body += f'\nfires on {100 * rate:.1f}% of states'
        self.tip.xy = (layer, ys[idx])
        self.tip.set_text(body)
        self.tip.set_visible(True)
        self.fig.canvas.draw_idle()
    # endregion


if __name__ == '__main__':
    viz = NetworkVisualizer()
    plt.show()
