"""Interactive one-input-at-a-time sweep viewer for the actor and critic.

This is ``test.py``'s picture driven by ``network_visualizer.py``'s controls.
Four panels, one per network input: each sweeps its own input across the full
slider range and plots the head's output along it.  The difference from
test.py is what the *other* three inputs are holding at - there they are pinned
at zero, here they sit wherever you left the dial and sliders, so every panel
is a slice through the state you are actually standing in.

Reading the picture
-------------------
  * Blue curve  - the output as that one input is swept, all others held.
  * Gold dot    - the current state, i.e. where the sliders put you on this
                  slice.  All four gold dots are the same state, so all four
                  sit at the same height; the curves around them are what would
                  happen if you moved that one input.
  * Orange dot  - the maximum along the sweep, annotated with its coordinates.
  * Dotted red  - the output at the current state, carried across the panel so
                  you can see which parts of the sweep beat where you are.

  The slope under the gold dot is the whole point: for the critic it is
  dV/d(input), the direction the value function wants that input to move; for
  the actor it is how hard the policy reacts to that input right now.

The 'others' radio switches the held inputs between the live slider values and
plain zeros, which is exactly test.py's view - useful as a reference frame when
the live slice starts looking strange.

Controls, networks and slots are imported from network_visualizer, so the two
tools always agree on ranges, checkpoints and the angle convention.

Run with:  python slice_visualizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

import neural_network as nn
from plot_utils import timestamp_figure
from network_visualizer import (AngleDial, INPUTS, NETWORKS, DIAL_INPUT,
                                ACTOR_FORCE, SAMPLES)


RESOLUTION = 400        # points per sweep; test.py's number, and plenty here

BG = '#1b1f27'
PANEL_BG = '#12151c'
FG = '#d8dde6'
MUTED = '#8d97a8'
CURVE = '#4c8fbd'
PEAK = '#ff9d5c'
HERE = '#f0d68c'
LEVEL = '#e06c6c'


class SliceVisualizer:

    def __init__(self, net_name='critic  (V)', slot=0):
        self.net_name = net_name
        self.slot = slot
        self.hold = 'others: live'
        self.values = np.zeros(len(INPUTS))
        self.net = None

        self.fig = plt.figure(figsize=(14, 9.0))
        self.fig.canvas.manager.set_window_title('NN slice viewer')
        timestamp_figure(self.fig, color='0.55')
        self.fig.patch.set_facecolor(BG)

        self._build_panels()
        self._build_controls()
        self.load_network()

    # region ----------------------------------------------------------- panels
    def _build_panels(self):
        """Four sweep axes in a 2x2 block, laid out over the controls."""
        self.axes = []
        self.curves = []
        self.peak_dots = []
        self.peak_txts = []
        self.here_dots = []
        self.here_lines = []     # vertical: where this input currently sits
        self.level_lines = []    # horizontal: the output at the current state
        for k in range(len(INPUTS)):
            col, row = k % 2, k // 2
            ax = self.fig.add_axes([0.065 + 0.495 * col, 0.665 - 0.305 * row,
                                    0.400, 0.245])
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=MUTED, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color('#39414f')
            ax.grid(alpha=0.18, color=MUTED, lw=0.6)

            short, caption, lo, hi = INPUTS[k]
            ax.set_xlim(lo, hi)
            ax.set_xlabel(f'input {k}: {short}   {caption}', color=MUTED, fontsize=8)
            ax.axvline(0, color='#5c6675', lw=0.8, ls=':', zorder=1)

            self.level_lines.append(ax.axhline(0, color=LEVEL, lw=0.8, ls=':',
                                               zorder=2))
            self.here_lines.append(ax.axvline(0, color=HERE, lw=0.8, ls='--',
                                              alpha=0.6, zorder=2))
            curve, = ax.plot([], [], color=CURVE, lw=2, zorder=3)
            peak, = ax.plot([], [], 'o', color=PEAK, ms=6, zorder=5)
            here, = ax.plot([], [], 'o', color=HERE, ms=7,
                            markeredgecolor=PANEL_BG, zorder=6)
            txt = ax.annotate('', xy=(0, 0), xytext=(0, 9),
                              textcoords='offset points', ha='center',
                              fontsize=7, color=PEAK)

            self.axes.append(ax)
            self.curves.append(curve)
            self.peak_dots.append(peak)
            self.peak_txts.append(txt)
            self.here_dots.append(here)

        self.title = self.fig.text(0.5, 0.955, '', ha='center', va='center',
                                   color=FG, fontsize=11)
        self.readout = self.fig.text(0.5, 0.925, '', ha='center', va='center',
                                     color=HERE, fontsize=9, family='monospace')
    # endregion

    # region --------------------------------------------------------- controls
    def _build_controls(self):
        # Same widget set and geometry as network_visualizer: the tilt input is
        # an AngleDial, the other three are sliders, each with a zero button.
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
                                valinit=0.0, color=CURVE)
                widget.label.set_fontsize(8)
                widget.label.set_color(FG)
                widget.valtext.set_color(FG)
                ax_z = self.fig.add_axes([0.585, y, 0.03, 0.028])
                zero_label = '0'
                row += 1

            widget.on_changed(self.on_slider)
            self.sliders[k] = widget

            btn = Button(ax_z, zero_label, color='#39414f', hovercolor='#4c5769')
            btn.label.set_color(FG)
            btn.label.set_fontsize(7 if zero_label != '0' else 8)
            btn.on_clicked(lambda _evt, w=widget: w.reset())
            self.zero_buttons[k] = btn   # keep a reference or it stops firing

        def panel(rect, labels, active, cb):
            ax_r = self.fig.add_axes(rect, facecolor='#2a2f3a')
            r = RadioButtons(ax_r, labels, active=active)
            for t in r.labels:
                t.set_color(FG)
                t.set_fontsize(8)
            r.on_clicked(cb)
            return r

        self.radio_net = panel([0.62, 0.145, 0.15, 0.085],
                               list(NETWORKS), list(NETWORKS).index(self.net_name),
                               self.on_network)
        self.radio_slot = panel([0.62, 0.025, 0.15, 0.10],
                                ['slot 0  (best)', 'slot 1  (prev)', 'slot 2  (periodic)'],
                                self.slot, self.on_slot)
        # Stands in for the activation/firing-rate switch: what the three
        # inputs a panel is *not* sweeping are held at.
        self.radio_hold = panel([0.80, 0.025, 0.17, 0.11],
                                ['others: live', 'others: zero'], 0, self.on_hold)

        ax_rand = self.fig.add_axes([0.80, 0.185, 0.08, 0.04])
        self.btn_rand = Button(ax_rand, 'random', color='#39414f', hovercolor='#4c5769')
        self.btn_rand.label.set_color(FG)
        self.btn_rand.on_clicked(self.on_random)

        ax_reset = self.fig.add_axes([0.89, 0.185, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'reset', color='#39414f', hovercolor='#4c5769')
        self.btn_reset.label.set_color(FG)
        self.btn_reset.on_clicked(self.on_reset)
    # endregion

    # region ------------------------------------------------------ net loading
    def load_network(self):
        cfg = NETWORKS[self.net_name]
        net = nn.NeuralNetwork(cfg['dim'], cfg['norm_fcn'], cfg['location'])
        try:
            net.theta_recover(self.slot)
        except FileNotFoundError:
            self.net = None
            self.title.set_text(f"no checkpoint in {cfg['location']} slot {self.slot}")
            self.title.set_color(LEVEL)
            self.readout.set_text('')
            for artist in (self.curves + self.peak_dots + self.here_dots):
                artist.set_data([], [])
            for txt in self.peak_txts:
                txt.set_text('')
            self.fig.canvas.draw_idle()
            return

        self.net = net
        self.is_actor = self.net_name.startswith('actor')
        self.out_label = 'motor force (N)' if self.is_actor else 'V'
        self.title.set_color(FG)
        for ax in self.axes:
            ax.set_ylabel(self.out_label, color=MUTED, fontsize=8)
        self._sample_range()
        self.update()

    def _sample_range(self):
        """A common y-window from random states, so panels stay comparable.

        Rescaling every panel to its own sweep would make a flat slice look as
        dramatic as a steep one.  One shared window over the state box the net
        is actually asked about keeps the shapes honest against each other, and
        the sweeps still get room because it is fitted to p1/p99, not min/max.
        """
        lows = np.array([lo for _, _, lo, _ in INPUTS])
        highs = np.array([hi for _, _, _, hi in INPUTS])
        X = np.random.uniform(lows, highs, size=(SAMPLES, len(INPUTS)))
        out = self.out_value(self.net.feedforward(X)[-1][:, 0])
        self.sampled = np.sort(out)
        lo = float(np.percentile(out, 1))
        hi = float(np.percentile(out, 99))
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        self.y_lo, self.y_hi = lo, hi

    def out_value(self, a):
        """Raw output activation -> the signed quantity it represents."""
        return (a - 0.5) * 2 * ACTOR_FORCE if self.is_actor else a

    def out_percentile(self, value):
        """How this output ranks against the sampled state box, in percent."""
        i = int(np.searchsorted(self.sampled, value))
        return 100.0 * i / len(self.sampled)
    # endregion

    # region -------------------------------------------------------- rendering
    def base_state(self):
        """What the three inputs a panel is not sweeping are pinned at."""
        return self.values if self.hold == 'others: live' else np.zeros(len(INPUTS))

    def sweep(self, index, n=RESOLUTION):
        """The head's output along one input axis, the rest held."""
        lo, hi = INPUTS[index][2], INPUTS[index][3]
        axis = np.linspace(lo, hi, n)
        X = np.tile(self.base_state(), (n, 1))
        X[:, index] = axis
        return axis, self.out_value(self.net.feedforward(X)[-1][:, 0])

    def update(self, _=None):
        if self.net is None:
            return

        base = self.base_state()
        here = float(self.out_value(
            self.net.feedforward(base.reshape(1, -1))[-1][0, 0]))

        sweeps = [self.sweep(k) for k in range(len(INPUTS))]
        # Widen the shared window if this state's slices run outside the box.
        lo = min(self.y_lo, min(float(v.min()) for _, v in sweeps))
        hi = max(self.y_hi, max(float(v.max()) for _, v in sweeps))
        span = max(hi - lo, 1e-9)

        for k, (axis, out) in enumerate(sweeps):
            self.curves[k].set_data(axis, out)

            peak = int(np.argmax(out))
            self.peak_dots[k].set_data([axis[peak]], [out[peak]])
            self.peak_txts[k].xy = (axis[peak], out[peak])
            self.peak_txts[k].set_text(f'max ({axis[peak]:.3f}, {out[peak]:.3f})')
            # A max sitting at either end of the sweep is common (the critic
            # often likes one extreme), and centred text there hangs off the
            # panel - so lean the label back inside near the edges.
            frac = (axis[peak] - axis[0]) / (axis[-1] - axis[0])
            self.peak_txts[k].set_ha('left' if frac < 0.12 else
                                     'right' if frac > 0.88 else 'center')

            x_here = base[k]
            self.here_dots[k].set_data([x_here], [here])
            self.here_lines[k].set_xdata([x_here, x_here])
            self.level_lines[k].set_ydata([here, here])
            # Extra headroom on top: the max label lives up there.
            self.axes[k].set_ylim(lo - 0.08 * span, hi + 0.18 * span)

        self._write_readout(base, here)
        self.fig.canvas.draw_idle()

    def _write_readout(self, base, here):
        held = 'sliders' if self.hold == 'others: live' else 'zero'
        self.title.set_text(f'{self.net_name}  slot {self.slot}   -   '
                            f'{self.out_label} with one input swept, '
                            f'others held at {held}')
        state = '  '.join(f'{INPUTS[k][0]}={base[k]:+.3f}' for k in range(len(INPUTS)))
        if self.is_actor:
            head = f'force = {here:+.2f} N'
        else:
            head = (f'V = {here:+.3f}   rated above '
                    f'{self.out_percentile(here):.0f}% of sampled states')
        self.readout.set_text(f'{state}      {head}')
    # endregion

    # region ------------------------------------------------------- callbacks
    def on_slider(self, _):
        self.values = np.array([s.val for s in self.sliders])
        self.update()

    def on_network(self, label):
        self.net_name = label
        self.load_network()

    def on_slot(self, label):
        self.slot = int(label.split()[1])
        self.load_network()

    def on_hold(self, label):
        self.hold = label
        self.update()

    def on_random(self, _):
        for s, (_, _, lo, hi) in zip(self.sliders, INPUTS):
            s.set_val(np.random.uniform(lo, hi))

    def on_reset(self, _):
        for s in self.sliders:
            s.set_val(0.0)
    # endregion


if __name__ == '__main__':
    viz = SliceVisualizer()
    plt.show()
