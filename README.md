# trippy_second

Simulating and balancing an **inverted pendulum on a cart** — with both classical (PID) and machine-learning controllers. This is the refactored second iteration of the [`trippy`](https://github.com/josephchiao/trippy) project.

## What it does

The project models a cart that slides along a horizontal rail with one or two rods ("pendulums") attached on top. Left alone, the rods swing down and flop around; the goal is to drive the cart's motor so the rods balance upright. It covers the full stack:

- **Physics engine** — the cart + double-pendulum equations of motion are derived symbolically with SymPy (Lagrangian mechanics) and integrated numerically with SciPy's `solve_ivp`.
- **Controllers** — a hand-tuned PID controller ("analog" control) and a neural-network controller trained with reinforcement learning.
- **Neural network** — a small feed-forward network implemented from scratch in NumPy (see `neural_network.py`), with saved weight sets in `nn_library/` and `nn_backup/`.
- **RL training** — a policy-gradient-style trainer (`RL_training.py`) that rewards keeping the pendulum upright and the cart near center.
- **Visualization** — Matplotlib animations of the cart, rods, and motor force, with a camera that pans to follow the cart.

Current status: single-pendulum balancing works, and double-pendulum balancing works in two separate modes under analog control.

## Repository layout

| File | Purpose |
|------|---------|
| `physics.py` | `DoublePendulum` class — symbolic derivation + numerical integration of the dynamics |
| `controller.py` | PID / control logic that decides the motor force each step |
| `neural_network.py` | From-scratch feed-forward neural network (NumPy) |
| `RL_training.py` | Reinforcement-learning trainer for the NN controller |
| `pid.py` | PID controller implementation |
| `theta_init.py` | Weight initialization helpers |
| `main.py` | Entry point — runs a simulation and renders the animation |
| `nn_library/`, `nn_backup/` | Saved network weights (`.npz`) |
| `*_legacy.py` | Earlier versions kept for reference |

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `sympy`, `matplotlib`

```bash
pip install numpy scipy sympy matplotlib
```

## Running it

```bash
python main.py
```

By default `main.py` runs the single-pendulum controller with the ML (neural-network) policy and shows the animation. Other run modes (e.g. `inverted_rod_1`, `inverted_rod_2`, `position_hold`) are available by editing the call at the bottom of `main.py`.

## Notes

This is a personal research/learning project and an active work in progress — expect rough edges and experimental code.
Total work in progress. Give me some time
5/11 update: Single pendulum balancing, and double pendulums balancing in two seprate modes achieved with analog control.
