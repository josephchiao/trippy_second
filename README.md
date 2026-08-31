# trippy_second

Simulating and balancing an **inverted pendulum on a cart** — with both classical (PID) and machine-learning controllers. This is the refactored second iteration of the [`trippy`](https://github.com/josephchiao/trippy) project.

## What it does

The project models a cart that slides along a horizontal rail with one or two rods ("pendulums") attached on top. Left alone, the rods swing down and flop around; the goal is to drive the cart's motor so the rods balance upright. It covers the full stack:

- **Physics engine** — the cart + pendulum equations of motion are derived symbolically with SymPy (Lagrangian mechanics), compiled to NumPy with `lambdify`, and integrated with a hand-rolled fixed-step RK4. Both a `SinglePendulum` and a `DoublePendulum` are available.
- **Controllers** — a hand-tuned PID controller ("analog" control) and a neural-network controller trained with reinforcement learning.
- **Neural network** — a small feed-forward network implemented from scratch in NumPy (see `neural_network.py`), with saved weight sets in `V_nn_library/` and `mu_nn_library/`.
- **RL training** — a one-step TD actor-critic (`RL_training.py`). A critic `NN_V` scores states, a frozen Polyak-averaged copy supplies the bootstrap, and an actor `NN_mu` outputs a sigmoid that is rescaled to a motor force. Exploration noise is itself learned, with an entropy bonus so it does not collapse.
- **Visualization** — a real-time interactive animation, a static post-run plot, and two tools for looking inside the trained networks.

Current status: single-pendulum balancing is solved under the learned policy, including recentering the cart. Double-pendulum balancing works in two separate modes under analog control. Current work is swing-up from hanging down.

## Repository layout

| File | Purpose |
|------|---------|
| `physics.py` | `SinglePendulum` / `DoublePendulum` — symbolic derivation + RK4 integration of the dynamics |
| `controller.py` | PID / control logic that decides the motor force each step (`analog` and `ML` modes) |
| `neural_network.py` | From-scratch feed-forward neural network (NumPy) |
| `theta_init.py` | Weight initialization helpers |
| `RL_training.py` | Actor-critic trainer — trains both networks together |
| `critict_trainer.py` | Critic-only trainer: fits `NN_V` to a fixed policy without touching `NN_mu` |
| `display.py` | **Real-time** animation with velocity readouts and a draggable target centre |
| `static_display.py` | Precomputes a run, then animates and/or plots it; also the double-pendulum runner |
| `network_visualizer.py` | Interactive activation viewer — which hidden units fire for a given state, and which are dead |
| `slice_visualizer.py` | Sweeps one network input at a time through the current state; shows the slope the policy/value is acting on |
| `test.py` | Same idea as `slice_visualizer`, non-interactive, other inputs pinned at zero |
| `pid.py` | PID controller implementation |
| `V_nn_library/`, `mu_nn_library/` | Live network weights (`.npz`) — critic and actor respectively |
| `V_nn_backup/`, `mu_nn_backup/` | Archived weight sets, each dated folder with a `note.txt` recording its training configuration |
| `*_legacy.py`, `*_backup.py` | Earlier versions kept for reference |

### Checkpoint slots

Both trainers write several numbered slots per network, and the visualizers read the same ones:

| Slot | Meaning |
|------|---------|
| `nn_theta_set_0.npz` | Current best |
| `nn_theta_set_1.npz` | Previous best — the rollback point if the policy collapses |
| `nn_theta_set_2.npz` | Periodic save, for recovering from a crash |
| `nn_theta_set_3.npz` | An episode that scored suspiciously well; saved but not trusted as a new best |

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `sympy`, `matplotlib`

```bash
pip install -r requirements.txt
```

## Running it

Train the actor and critic together:

```bash
python RL_training.py
```

Live diagnostics plot per episode: reward, exploration noise (`log_std`), and mean advantage. Reward dots are coloured by whether the episode survived the full runtime.

> **Note:** `RL_trainer.__init__` calls `theta_generate()`, which **wipes** `V_nn_library/` and `mu_nn_library/` before loading. Comment those two lines out to resume from existing weights instead of starting fresh, and copy anything you want to keep into the `*_backup/` folders first.

Refine only the critic against a policy you are happy with:

```bash
python critict_trainer.py
```

Watch a trained policy run in real time — drag the slider, or click the track, to move the target centre while it balances:

```bash
python display.py
```

Precompute a run and inspect it as an animation plus static plots. Edit the call at the bottom of the file to pick the controller (`ML`, `inverted_rod`, `position_hold`, `None`), the checkpoint slot, and the initial state; `DP_run` and `custom_run` do the same for the double pendulum:

```bash
python static_display.py
```

Look inside the networks:

```bash
python network_visualizer.py   # which units are on, near switching, or dead
python slice_visualizer.py     # output vs. one input, sliced through the current state
```

Angle convention throughout: `theta = pi` is upright, `theta = 0` is hanging down, positive `theta` is counter-clockwise, and positive motor force pushes the cart right.

## Notes

This is a personal research/learning project and an active work in progress — expect rough edges and experimental code.
Total work in progress. Give me some time

- **5/11 update:** Single pendulum balancing, and double pendulums balancing in two seprate modes achieved with analog control.
- **8/29 update:** Single pendulum balancing achieved with the learned policy. Critic recalibrated separately with `critict_trainer.py`; network visualizers added to see which units actually fire.
- **8/30 update:** Actor retrained on the calibrated critic — the cart now recenters to within 0.004. Real-time display with a movable target centre added.
- **8/31 update:** `RL_training.py` moved to swing-up from hanging down. Restored the episode terminal condition, fixed the checkpoint/plot logic that assumed positive rewards, and added NaN/inf guards to the reward, the rollout, every weight update, and checkpoint loads.
