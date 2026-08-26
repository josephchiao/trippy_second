"""Actor-critic trainer for balancing a single inverted pendulum on a cart.

The setup is a one-step TD (advantage) actor-critic:

  * Critic  ``NN_V``   maps a state to its estimated value.  A frozen copy
    ``NN_V_tgt`` is Polyak-averaged toward the live critic and supplies the
    bootstrap term, which keeps the TD target from chasing its own tail.
  * Actor   ``NN_mu``  outputs a sigmoid in [0, 1] that is rescaled to a motor
    force in [-100, 100] N.  Exploration comes from Gaussian noise whose scale
    ``exp(log_std)`` is itself learned (with an entropy bonus so it does not
    collapse to zero).

State vector throughout is ``[cart position, pendulum angle, cart velocity,
angular velocity]``, with angle = pi meaning upright.  Gradients are collected
into mini-batches and applied every ``batch_size`` frames.
"""

import neural_network as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
import random
import physics
from datetime import datetime

init_log_std = 0  # Initial exploration noise level (log scale); exp(0) = 1 N of noise


class RL_trainer:
    """Owns the actor, the critic, its target copy, and the training loop."""

    def __init__(self, model):

        self.model = model            # physics.SinglePendulum instance to train against
        self.log_std = init_log_std   # Log of the exploration noise std dev (learned)
        self.log_floor = -3           # Clamps on log_std so exploration can neither
        self.log_ceiling = 3          # vanish nor blow the action range apart
        self.d_log_std = 0            # Accumulated log_std gradient for the current batch
        self.excess_advantage_count = 0  # Count of frames with advantage magnitude > 2, which are ignored to avoid destabilizing the batch
        self.NN_V = nn.NeuralNetwork((4, 64, 64, 1), [nn.ReLU, nn.ReLU, nn.linear], 'V_nn_library')
        self.NN_mu = nn.NeuralNetwork((4, 16, 16, 1), [nn.ReLU, nn.ReLU, nn.sigmoid], 'mu_nn_library')

        # Uncomment these two to wipe the checkpoint folders and start from
        # freshly initialized weights instead of resuming a previous run.
        # self.NN_V.theta_generate()
        # self.NN_mu.theta_generate()

        self.NN_V.theta_recover()
        self.NN_mu.theta_recover()

        # Target critic: same architecture, weights seeded from the live critic.
        self.NN_V_tgt = nn.NeuralNetwork((4, 64, 64, 1), [nn.ReLU, nn.ReLU, nn.linear], 'V_nn_library')
        self.sync_target(hard=True)

    def reward(self, state, episode = 0):

        """Max reward should be 1. Reward is based on how upright the pendulum is and how close the cart is to the center."""

        location_cf = 0.05      # Weight on staying near x = 0
        spl_location_cf = 0   # Weight on staying near x = 0 when the cart is far away
        angle_cf = 0.9          # Weight on staying upright (angle = pi)
        time_reward = 0       # Flat per-frame bonus; positive rewards survival, negative penalizes stalling
        effort_reward = 0.05

        # -cos(angle) peaks at +1 upright and bottoms at -1 hanging down.
        # The location term decays linearly with |x| and is floored at 0 so a
        # far-away cart is merely unrewarded rather than heavily punished.
        
        # reward = time_reward - angle_cf * math.cos(state[1]) + max(0, location_cf * (1 - 0.3 * abs(state[0])))
        reward = time_reward - angle_cf * math.cos(state[1]) + location_cf * 0.5 / (0.5 + abs(state[0])) + spl_location_cf * (abs(state[0]) < 0.1) - effort_reward * (self.model.motor_force / 100) ** 2   # A hyperbolic decay that is smooth and never hits zero
        
        return reward

    def normalize(self, state):
        """Normalize terms for the state to be readable by the neural network.

        Currently a pass-through: the early ``return`` bypasses normalization.
        Delete that line to re-enable the squashing below, which maps the
        unbounded cart position into (-1, 1) and leaves the other three
        components untouched.
        """
        return state
        return np.array([2/(1 + np.exp(-2 * state[0])) - 1, state[1], state[2], state[3]])

    def sync_target(self, hard=False, tau=0.05):
        """Blend the live critic into the frozen target. hard=True copies outright."""
        if hard:
            self.NN_V_tgt.theta = [w.copy() for w in self.NN_V.theta]
            self.NN_V_tgt.b     = [w.copy() for w in self.NN_V.b]
        else:
            # Polyak averaging: the target creeps toward the live critic at rate tau.
            for i in range(len(self.NN_V.theta)):
                self.NN_V_tgt.theta[i] = (1 - tau) * self.NN_V_tgt.theta[i] + tau * self.NN_V.theta[i]
                self.NN_V_tgt.b[i]     = (1 - tau) * self.NN_V_tgt.b[i]     + tau * self.NN_V.b[i]

    def backward_std(self, action, mu, sigma, advantage):
        """Single-frame policy gradient. Superseded by backward_std_MC; kept for reference.

        Returns dL/dmu and accumulates dL/dlog_std into self.d_log_std, using the
        standard Gaussian log-likelihood gradients scaled by the advantage.
        """
        epsilon = 1e-12  # Small constant to prevent division by zero
        action_discrepency = action / 200 + 0.5 - mu  # Realized action minus the mean, in mu's [0, 1] units

        d_mu = -advantage * (action_discrepency / (sigma ** 2 + epsilon))
        # d_mu = -advantage * action_discrepency  # Un-normalized variant, ignores sigma

        # Entropy-free log_std gradient: positive advantage with a larger-than-typical
        # deviation widens sigma, a smaller-than-typical one narrows it.
        step_d_log_std = -advantage * ((action_discrepency**2 / (sigma ** 2 + epsilon)) - 1.0)

        # Optional clip if a single frame ever produces an explosive log_std step.
        # if abs(step_d_log_std) > 20.0:
        #     step_d_log_std = np.sign(step_d_log_std) * 20.0

        self.d_log_std += step_d_log_std

        return d_mu   

    def backward_std_MC(self, actions, mus, sigmas, advantages):
        """Batched policy gradient over one mini-batch of frames.

        Note ``actions`` is not the raw motor force: callers pass the *action
        discrepancy* (realized action minus mu, in mu's [0, 1] units), which is
        why it appears directly in the gradients below rather than as (a - mu).
        ``mus`` is unused here and kept only to keep call sites symmetric.

        Returns the per-frame dL/dmu array and accumulates dL/dlog_std for the batch.
        """
        epsilon = 1e-12  # Small constant to prevent division by zero

        calibrated_advantages = advantages - np.mean(advantages)  # Center the advantages to have a mean of zero

        d_mus = -calibrated_advantages * (actions / (sigmas ** 2 + epsilon))
        step_d_log_std = -calibrated_advantages * ((actions**2 / (sigmas ** 2 + epsilon)) - 1.0)

        self.d_log_std += np.sum(step_d_log_std)

        return d_mus
    
    def train(self, max_runtime = 60):
        """Run the training loop indefinitely, plotting progress live.

        max_runtime is the episode time cap in seconds; an episode that reaches
        it is treated as truncated (still bootstrapped) rather than failed.
        variance is currently unused.
        """

        t = 0
        solution = []
        state_history = []
        total_cost = 0


        gamma = 0.999  # Discount factor (how much we care about the future)
        reward_history = []
        log_std_history = []
        advantage_history = []
        signed_advantage_history = []
        fail_count = 0  # Consecutive poor episodes; triggers rollback once high enough

        start_time = datetime.now()

        #region display setup
        # --- Live diagnostics: reward, exploration noise, and advantage per episode ---
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)
        fig.suptitle(f'Run started: {start_time.strftime("%Y-%m-%d %H:%M:%S")}', fontsize=10)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Log Std')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Mean Advantage')
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax3.grid(True, alpha=0.3)
        line1, = ax1.plot([], [])
        line2, = ax2.plot([], [])
        line3, = ax3.plot([], [], label='|advantage|')  # Magnitude: how wrong the critic is
        line4, = ax3.plot([], [], label='signed')       # Sign: whether it over- or under-estimates
        ax3.legend(loc='upper right', fontsize='small')
        # endregion

        best_reward = 0
        second_best_reward = 0
        previously_saved = False  # don't let recovery load a stale checkpoint from a previous run
        learning_rate_discount = 1  # Actor-side LR scale, annealed as average reward climbs

        for episode in range(1000000):

            learning_rate = 0.0001
            # normal learning rate = 0.0001

            # # Warm-up variant: let the critic race ahead early, then even out.
            # if episode < 1000:
            #     V_lrn = 50  # Critic learning rate multiplier
            # else:
            #     V_lrn = 3  # Critic learning rate multiplier

            V_lrn = 10  # Critic learns faster than the actor so its targets stay ahead of the policy
    

            # random_angle = np.pi/30   # Fixed tilt off vertical at episode start
            random_angle = 0
            starting_location = 1.5     # Fixed cart offset from center at episode start
            
            total_episode_reward = 0

            # Shared buffer across both sides so each batch mixes both starting configs.
            # Randomized order removes the systematic bias of always training side -1 first.
            states_memory = []
            target_V_memory = []
            target_mu_memory = []

            mu_memory = []    # Actor means, per frame
            disc_memory = []  # Action discrepancies (realized action - mu), per frame
            adv_memory = []   # Clipped advantages, per frame
            self.d_log_std = 0
            runtimes = []           # Frames survived on each side
            episode_advantages = [] # Unclipped advantages, for the diagnostics plot only
            sides = [-1, 1]
            random.shuffle(sides)

            # Entropy bonus coefficient: high while reward is low to keep exploring,
            # tapering toward zero (and slightly negative) as the policy gets good.
            # if episode < 100:
            #     dynamic_entropy = 0.05  # Fixed bonus until there is enough reward history to average
            # else:       
            dynamic_entropy = max(-0.001, (-np.mean(reward_history[-100:])/2000 + 1) * 0.05)

            # if episode < 300:
            #     learning_rate_discount = 0
            # elif episode < 600:
            #     learning_rate_discount = episode / 600 - 1

            for side in sides:
                # Start tilted and offset toward `side`, so both mirror images get trained.
                self.model.state = [side * starting_location, np.pi + side * random_angle, 0, 0]
                t = 0
                done = False
                while not done:
                    t += 1

                    # Normalize before asking the networks for a value and an action
                    normalized_state = self.normalize(self.model.state)

                    V = self.NN_V.feedforward(normalized_state)[-1][0][0]
                    mu = self.NN_mu.feedforward(normalized_state)[-1][0][0]
                    # mu is a sigmoid in [0, 1]; recenter and scale to +/-100 N, then add exploration noise.
                    self.model.motor_force = (mu - 0.5) * 200 + np.exp(self.log_std) * np.random.randn()

                    next_state = self.model.rk4_step()
                    reward = self.reward(next_state, episode=episode)
                    total_episode_reward += reward
                    # Fail conditions: pendulum past horizontal, time cap hit, velocities or
                    # cart position running away, or the integrator producing NaNs.
                    done = next_state[1] <= np.pi/2 or next_state[1] >= 3*np.pi/2 or t >= (max_runtime) * self.model.refresh_rate or abs(next_state[2]) > 100 or abs(next_state[3]) > 100 or abs(next_state[0]) > 5 or np.isnan(next_state).any()  # Check for failure conditions 

                    normalized_next_state = self.normalize(next_state)
                    next_critic = self.NN_V_tgt.feedforward(normalized_next_state)[-1][0][0]

                    if done and t < (max_runtime) * self.model.refresh_rate:
                        target_value = reward                             # advance terminal (no future)
                    else:
                        target_value = reward + gamma * next_critic       # alive, or truncated at timeout

                    advantage_unclipped = target_value - V
                    episode_advantages.append(advantage_unclipped)
                    advantage = np.clip(advantage_unclipped, -5.0, 5.0)  # Keep one bad frame from dominating the batch

                    # --- THE BACKWARD PASS ---
                    # Critic error (Mean Squared Error):
                    # Loss = 0.5 * (target_value - critic)^2
                    # dL/dV = -(target_value - critic) = -advantage
                    d_V = -advantage

                    # Per-frame actor update, replaced by the batched backward_std_MC below.
                    # d_mu = self.backward_std(
                    #     action=self.model.motor_force,
                    #     mu=mu,
                    #     sigma=np.exp(self.log_std)/200,
                    #     advantage=advantage)

                    # Move to the next frame
                    self.model.state = next_state

                    states_memory.append(normalized_state)
                    target_V = V + advantage_unclipped  # Use unclipped advantage for the Critic target to avoid biasing the Critic towards underestimating the value of states 
                    target_V = np.clip(target_V, -200, 2000)  # Bound the target to the reachable return range
                    target_V_memory.append(target_V)

                    mu_memory.append(mu)
                    disc_memory.append(self.model.motor_force / 200 + 0.5 - mu)  # Realized action minus mu, in [0, 1] units
                    adv_memory.append(advantage)


                    batch_size = 240 

                    # Flush the batch mid-episode once enough frames have accumulated.
                    if len(states_memory) >= batch_size:

                        if abs(np.mean(adv_memory)) > 2.0:
                            self.excess_advantage_count += 1
                        else:
                            d_mu = self.backward_std_MC(
                                actions=np.array(disc_memory),
                                mus=np.array(mu_memory),
                                sigmas=np.exp(self.log_std)/200,
                                advantages=np.array(adv_memory))

                            # The network trains toward a target, so express the gradient
                            # step as "where mu should have been" rather than a delta.
                            target_mu_memory = np.array(mu_memory) - d_mu

                            self.NN_mu.backward(np.array(states_memory), np.array(target_mu_memory).reshape(-1, 1), learning_rate_discount * learning_rate / batch_size)
                            # Update exploration noise; subtracting dynamic_entropy pushes
                            # log_std back up so it resists collapsing to the floor.
                            self.log_std -= learning_rate_discount * learning_rate * (self.d_log_std / batch_size - dynamic_entropy)
                            self.log_std = np.clip(self.log_std, self.log_floor, self.log_ceiling)

                        self.NN_V.backward(np.array(states_memory), np.array(target_V_memory).reshape(-1, 1),  learning_rate * V_lrn / batch_size)


                        states_memory = []
                        target_V_memory = []
                        target_mu_memory = []
                        disc_memory = []
                        adv_memory = []
                        mu_memory = []
                        self.d_log_std = 0

                runtimes.append(t)

            # Flush any remaining experience after both sides complete
            if len(states_memory) > 16: # Only flush if we have a reasonable amount of data to avoid overfitting to a tiny batch

                if abs(np.mean(adv_memory)) > 2.0:
                    self.excess_advantage_count += 1
                else:
                    d_mu = self.backward_std_MC(
                        actions=np.array(disc_memory),
                        mus=np.array(mu_memory),
                        sigmas=np.exp(self.log_std)/200,
                        advantages=np.array(adv_memory))

                    target_mu_memory = np.array(mu_memory) - d_mu

                    # NOTE: the network updates divide by batch_size while the log_std update
                    # divides by the actual number of leftover frames.
                    self.NN_mu.backward(np.array(states_memory), np.array(target_mu_memory).reshape(-1, 1), learning_rate_discount * learning_rate / batch_size)
                    self.log_std -= learning_rate_discount * learning_rate * (self.d_log_std / len(states_memory) - dynamic_entropy)
                    self.log_std = np.clip(self.log_std, self.log_floor, self.log_ceiling)


                self.NN_V.backward(np.array(states_memory), np.array(target_V_memory).reshape(-1, 1), learning_rate * V_lrn / batch_size)


            print(f"Episode {episode} finished! Total Reward: {total_episode_reward:.2f}, runtime = {runtimes[0]}, {runtimes[1]}, excess_advantage_count = {self.excess_advantage_count}")
            self.excess_advantage_count = 0  # Reset for the next episode

            # --- Checkpointing ---
            # Slot 0 = current best, slot 1 = previous best (rollback point),
            # slot 2 = periodic crash save, slot 3 = suspicious-outlier save.
            if episode == 0:
                best_reward = total_episode_reward
                second_best_reward = total_episode_reward
            
            elif total_episode_reward >= best_reward and total_episode_reward <= best_reward * 2 or episode == 0:  # Only consider it a new best if it's not an outlier that might be a lucky fluke
                second_best_reward = best_reward
                best_reward = total_episode_reward
                self.NN_mu.theta_backup()  # slot0 (old best) -> slot1 before overwriting
                self.NN_V.theta_backup()  # slot0 (old best) -> slot1 before overwriting
                self.NN_mu.theta_save()    # current weights -> slot0
                self.NN_V.theta_save()    # current weights -> slot0
                previously_saved = True
                print('Saved to 0!')

            elif total_episode_reward >= second_best_reward and total_episode_reward < best_reward * 2:  # Only update second best if it's not an outlier that might be a lucky fluke
                second_best_reward = total_episode_reward
                self.NN_mu.theta_save()    # save without touching slot1 backup
                self.NN_V.theta_save()    # save without touching slot1 backup
                previously_saved = True
                print('Saved to 0 (Second best)!')
            
            if total_episode_reward > best_reward * 2:  # Too good to trust: keep the weights in slot3 but don't let the outlier move the best/second-best records
                self.NN_mu.theta_save(3)  # save to slot3 for recovery, not policy collapse
                self.NN_V.theta_save(3)  # save to slot3 for recovery, not policy collapse
                total_episode_reward = best_reward  # Clamp for the plot so one fluke doesn't flatten the reward curve's scale
                print('Saved to 3!')
            
            if episode % 100 == 0 and fail_count == 0:  # Only save periodically if we're not in a failure streak
                self.NN_mu.theta_save(2)  # periodic save to slot2 for recovery from crashes, not policy collapse
                self.NN_V.theta_save(2)  # periodic save to slot2 for recovery from crashes, not policy collapse
                print('Periodic Save to 2!')

            # --- Policy collapse detection ---
            if total_episode_reward < max(20, second_best_reward * 0.2):  # An episode this bad counts as a failure; the max() keeps the bar meaningful early on
                fail_count += 1
            else:
                fail_count = 0  # Any decent episode proves the policy is still viable

            if fail_count >= 200 and previously_saved and False:  # A long failure streak means the policy really has collapsed: roll back
                self.NN_mu.theta_recover(i = 1)
                self.NN_V.theta_recover(i = 1)
                self.sync_target(hard=True)  # Target would otherwise still hold the collapsed critic
                print('policy_collapse')
                self.log_std = init_log_std  # restore exploration, not kill it
                fail_count = 0

            self.sync_target(tau=0.05)
            reward_history.append(total_episode_reward)
            log_std_history.append(self.log_std)
            advantage_history.append(np.mean(np.abs(episode_advantages)) if episode_advantages else 0.0)
            signed_advantage_history.append(np.mean(episode_advantages) if episode_advantages else 0.0)

            # Anneal the actor's learning rate as the running average approaches the
            # 7200-reward ceiling (2 sides * 60 s * 60 fps at ~1 reward per frame).
            recent = np.mean(reward_history[-200:])
            learning_rate_discount = float(np.clip(3 - 3 * recent / 7200, 0.1, 1.0))

            #region Live plot update
            # Redraw the diagnostics with this episode's data appended
            episodes = range(len(reward_history))
            line1.set_data(episodes, reward_history)
            line2.set_data(episodes, log_std_history)
            line3.set_data(episodes, advantage_history)
            line4.set_data(episodes, signed_advantage_history)
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            ax3.relim()
            ax3.autoscale_view()
            fig.canvas.flush_events()
            #endregion 


        # Only reached if the episode loop is broken out of; leaves the plot up.
        self.NN_mu.theta_save(2)
        self.NN_V.theta_save(2)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    y0 = [0, np.pi, 0, 0]  # Start centered and upright: [x, angle, x_dot, angle_dot]
    SP = physics.SinglePendulum(params=(9.8, 1, 1, 1), y0 = y0, refresh_rate=60)  # g, cart mass, rod mass, rod length
    main = RL_trainer(SP)

    main.train(max_runtime = 30)  # seconds
