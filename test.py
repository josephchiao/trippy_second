import numpy as np
import matplotlib.pyplot as plt
import neural_network as nn


# --- Network inputs, in the order normalize() feeds them ---
# state = [x, theta, x_dot, theta_dot]; normalize() shifts theta by -pi.
INPUT_NAMES = ['x (cart position)',
               'theta - pi (angle offset)',
               'x_dot (cart velocity)',
               'theta_dot (angular velocity)']

# Sweep range for each input (others are held at zero).
INPUT_RANGES = [(-3.0, 3.0),
                (-np.pi, np.pi),
                (-10.0, 10.0),
                (-10.0, 10.0)]

RESOLUTION = 400


def V_sweep(NN_V, index, lo, hi, n = RESOLUTION):
    """V evaluated along one input axis, all other inputs pinned at zero."""
    sweep = np.linspace(lo, hi, n)
    X = np.zeros((n, NN_V.dim[0]))
    X[:, index] = sweep
    V = NN_V.feedforward(X)[-1][:, 0]
    return sweep, V


def plot_V_slices(NN_V):
    fig, axes = plt.subplots(2, 2, figsize = (11, 8))
    fig.suptitle('Critic V, one input swept at a time (all others = 0)')

    for index, ax in enumerate(axes.flat):
        lo, hi = INPUT_RANGES[index]
        sweep, V = V_sweep(NN_V, index, lo, hi)

        ax.plot(sweep, V, color = 'tab:blue', lw = 2)

        peak = int(np.argmax(V))
        x_peak, V_peak = sweep[peak], V[peak]
        ax.plot(x_peak, V_peak, 'o', color = 'tab:orange', ms = 7, zorder = 5,
                label = 'max')
        ax.annotate(f'({x_peak:.3f}, {V_peak:.3f})',
                    xy = (x_peak, V_peak),
                    xytext = (0, 10), textcoords = 'offset points',
                    ha = 'center', fontsize = 8, color = 'tab:orange')

        ax.axvline(0, color = 'k', lw = 0.8, ls = ':')
        ax.axhline(NN_V.feedforward(np.zeros((1, NN_V.dim[0])))[-1][0, 0],
                   color = 'tab:red', lw = 0.8, ls = ':', label = 'V at origin')
        ax.set_xlabel(f'input {index}: {INPUT_NAMES[index]}')
        ax.set_ylabel('V')
        ax.grid(alpha = 0.3)
        ax.legend(fontsize = 8)

    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    NN_V = nn.NeuralNetwork((4, 64, 64, 1), [nn.ELU, nn.ELU, nn.linear], 'V_nn_library')
    NN_V.theta_recover(2)
    plot_V_slices(NN_V)
