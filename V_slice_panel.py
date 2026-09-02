"""The critic V slice panel: V swept along each input axis, one input at a time.

Same content as test.plot_V_slices, but the figure is built once and updated in
place, so a training run keeps one extra window rather than opening a new one on
every redraw. The previous `trail` curves stay on each panel, fading with age, so
drift in the critic is visible at a glance.

Shared by RL_training.py and critict_trainer.py.
"""

from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import test as critic_slices  # the V-slice sweep/plot definitions
from plot_utils import timestamp_figure

trail = 20  # How many previous slice curves stay on the plot, fading out
# Newest ghost first: the older a curve is, the fainter it is drawn. Generated so
# the ramp still spans the full fade range whatever trail is set to.
alphas = np.linspace(0.45, 0.05, trail)


class VSlicePanel:
    """One slice window. Call update() with the critic whenever a redraw is due."""

    def __init__(self, title = 'critic V slices'):
        self.title = title
        self.fig = None                            # built lazily by update()
        self.history = deque(maxlen = trail)       # the last few sets of curves

    def build(self):
        """The 2 x 2 panel, with empty artists to fill later."""
        fig, axes = plt.subplots(2, 2, figsize = (11, 8), constrained_layout = True)
        fig.canvas.manager.set_window_title(self.title)
        # Titled here so the first draw is already complete; the episode number is
        # appended on each redraw below.
        fig.suptitle('Critic V, one input swept at a time (all others = 0)')
        timestamp_figure(fig)

        self.fig = fig
        self.axes = list(axes.flat)
        self.artists = []
        for index, ax in enumerate(self.axes):
            # Ghosts of earlier redraws, drawn under the current curve and fading
            # with age.
            ghosts = []
            for age in range(trail):
                ghost, = ax.plot([], [], color = 'tab:blue', lw = 1, zorder = 1,
                                 alpha = alphas[age],
                                 label = f'previous {trail}' if age == 0 else None)
                ghosts.append(ghost)
            line, = ax.plot([], [], color = 'tab:blue', lw = 2, zorder = 3)
            peak, = ax.plot([], [], 'o', color = 'tab:orange', ms = 7, zorder = 5,
                            label = 'max')
            note = ax.annotate('', xy = (0, 0), xytext = (0, 10),
                               textcoords = 'offset points', ha = 'center',
                               fontsize = 8, color = 'tab:orange')
            ax.axvline(0, color = 'k', lw = 0.8, ls = ':')
            origin = ax.axhline(0.0, color = 'tab:red', lw = 0.8, ls = ':',
                                label = 'V at origin')
            # The sweep range is fixed, so only the vertical scale is rescaled below.
            ax.set_xlim(*critic_slices.INPUT_RANGES[index])
            ax.set_xlabel(f'input {index}: {critic_slices.INPUT_NAMES[index]}')
            ax.set_ylabel('V')
            ax.grid(alpha = 0.3)
            ax.legend(fontsize = 8)
            self.artists.append((line, peak, note, origin, ghosts))

    def apply_trail(self):
        """Fill the ghost lines from the stored history, and rescale around them."""
        for index, (ax, artists) in enumerate(zip(self.axes, self.artists)):
            ghosts = artists[4]
            for age, ghost in enumerate(ghosts):
                # The trail holds whole redraws, newest first, so ghost `age`
                # reads straight off the deque. Early on there are fewer redraws
                # than ghosts, and the spare ones stay empty.
                if age < len(self.history):
                    ghost.set_data(*self.history[age][index])
                else:
                    ghost.set_data([], [])
            ax.relim()
            ax.autoscale_view(scalex = False)

    def update(self, NN_V, episode):
        """Redraw the critic along each input axis, all other inputs pinned at zero.

        Closing the window is fine - the next call rebuilds it (the trail survives).
        """
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.build()

        V_origin = NN_V.feedforward(np.zeros((1, NN_V.dim[0])))[-1][0, 0]
        curves = []
        for index, (ax, artists) in enumerate(zip(self.axes, self.artists)):
            line, peak, note, origin, _ghosts = artists
            lo, hi = critic_slices.INPUT_RANGES[index]
            sweep, V = critic_slices.V_sweep(NN_V, index, lo, hi)
            curves.append((sweep, V))

            line.set_data(sweep, V)
            top = int(np.argmax(V))
            peak.set_data([sweep[top]], [V[top]])
            note.xy = (sweep[top], V[top])
            note.set_text(f'({sweep[top]:.3f}, {V[top]:.3f})')
            origin.set_ydata([V_origin, V_origin])

        # Ghosts show the redraws before this one, so the trail is drawn (and the
        # axes rescaled around it) before this episode's curves join the history.
        self.apply_trail()
        self.history.appendleft(curves)
        self.fig.suptitle('Critic V, one input swept at a time '
                          f'(all others = 0) - episode {episode}')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
