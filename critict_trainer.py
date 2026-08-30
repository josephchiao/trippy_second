import neural_network as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
import random
import physics
from datetime import datetime
 

class RL_trainer:

    def __init__(self, model):
        
        self.model = model
        self.NN_V = nn.NeuralNetwork((4, 64, 64, 1), [nn.ELU, nn.ELU, nn.linear], 'V_nn_library')
        self.NN_mu = nn.NeuralNetwork((4, 16, 16, 1), [nn.ELU, nn.ELU, nn.sigmoid], 'mu_nn_library')

        self.NN_V.theta_recover()
        self.NN_mu.theta_recover()

        self.NN_V_tgt = nn.NeuralNetwork((4, 64, 64, 1), [nn.ELU, nn.ELU, nn.linear], 'V_nn_library')
        self.sync_target(hard=True)

    def reward(self, state, episode = 0, next_phase = False):

        """Max reward should be 1. Reward is based on how upright the pendulum is and how close the cart is to the center."""

        # schedule_scaler = 0.15 * np.clip((episode - 300) / 2000, 0, 1) 

        location_cf = 0.1       # Weight on staying near x = 0
        spl_location_cf = 0   # Weight on staying near x = 0 when the cart is far away
        angle_cf = 1.8         # Weight on staying upright (angle = pi)
        time_reward = -1       # Flat per-frame bonus; positive rewards survival, negative penalizes stalling
        effort_reward = 0
        velocity_cf = 0.1
        # -cos(angle) peaks at +1 upright and bottoms at -1 hanging down.
        # The location term decays linearly with |x| and is floored at 0 so a
        # far-away cart is merely unrewarded rather than heavily punished.
        
        # reward = time_reward - angle_cf * math.cos(state[1]) + max(0, location_cf * (1 - 0.3 * abs(state[0])))
        reward = time_reward - angle_cf * math.cos(state[1]) + location_cf * 0.25 / (0.25 + abs(state[0])) - velocity_cf * np.clip(state[2]**2 / 16.0 + state[3]**2, 0, 1) + spl_location_cf * (abs(state[0]) < 0.1) + effort_reward * (1 - abs(np.tanh(self.model.motor_force / 3)))
        
        return reward

    def normalize(self, state):
        """Normalize terms for the state to be readable by the neural network."""
        return np.array([state[0], state[1]-np.pi, state[2], state[3]])


    def sync_target(self, hard=False, tau=0.05):
            """Blend the live critic into the frozen target. hard=True copies outright."""
            if hard:
                self.NN_V_tgt.theta = [w.copy() for w in self.NN_V.theta]
                self.NN_V_tgt.b     = [w.copy() for w in self.NN_V.b]
            else:
                for i in range(len(self.NN_V.theta)):
                    self.NN_V_tgt.theta[i] = (1 - tau) * self.NN_V_tgt.theta[i] + tau * self.NN_V.theta[i]
                    self.NN_V_tgt.b[i]     = (1 - tau) * self.NN_V_tgt.b[i]     + tau * self.NN_V.b[i]

    def train(self, variance = 0, max_runtime = 60):

        t = 0
        solution = []
        state_history = []
        total_cost = 0


        gamma = 0.99  # Discount factor (how much we care about the future)
        reward_history = []
        log_std_history = []
        advantage_history = []
        signed_advantage_history = []
        clip_rate_history = []   # Fraction of frames per episode where the advantage clip bound
        best_episodes = []       # Episode indices where a new best reward was set
        best_rewards = []        # The rewards at those episodes, for the plot highlight
        eval_episodes = []       # Episodes where the deterministic (no-noise) evaluation ran
        eval_rewards = []        # Total reward of that evaluation, for the plot overlay
        full_sides_history = []  # Per episode: how many of the two sides survived the full runtime
        fail_count = 0  # Consecutive poor episodes; triggers rollback once high enough
        fail_count = 0

        #region display setup
        # --- Live diagnostics: reward, exploration noise, and advantage per episode ---
        plt.ion()
        start_time = datetime.now()
        fig, (ax1, ax3) = plt.subplots(2, 1, constrained_layout=True)
        fig.suptitle(f'Run started: {start_time.strftime("%Y-%m-%d %H:%M:%S")}', fontsize=10)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Mean Advantage')
        ax1.grid(True, alpha=0.3)
        ax3.grid(True, alpha=0.3)
        line1, = ax1.plot([], [], color='0.75', lw=1, zorder=1)
        # Per-episode reward dots, colored by how many of the two starting sides lasted
        # the whole runtime: both (blue), only one (yellow), neither (red).
        marker_both, = ax1.plot([], [], linestyle='none', marker='o', markersize=1,
                                color='tab:blue', label='both sides full', zorder=2)
        marker_one, = ax1.plot([], [], linestyle='none', marker='o', markersize=1,
                               color='gold', label='one side full', zorder=2)
        marker_neither, = ax1.plot([], [], linestyle='none', marker='o', markersize=1,
                                   color='tab:red', label='neither side full', zorder=2)
        # Stars mark the episodes that set a new best reward; the dashed line is the
        # running best, so a flat stretch reads as "no progress since here".
        line_best, = ax1.plot([], [], linestyle='--', drawstyle='steps-post', color='0.6', lw=1, zorder=2)
        marker_best, = ax1.plot([], [], linestyle='none', marker='*', markersize=7,
                                color='tab:orange', label='new max', zorder=5)
        # Deterministic evaluation run every 100 episodes (both starting sides summed)
        marker_eval, = ax1.plot([], [], linestyle='-', marker='o', markersize=3,
                                color='tab:green', lw=1, label='eval (no noise)', zorder=4)
        ax1.legend(loc='lower right', fontsize='small', ncol=2)
        line3, = ax3.plot([], [], label='|advantage|')  # Magnitude: how wrong the critic is
        line4, = ax3.plot([], [], label='signed')       # Sign: whether it over- or under-estimates
        ax3.legend(loc='upper right', fontsize='small')
        # endregion

        best_reward = 0
        second_best_reward = 0
        previously_saved = False  # don't let recovery load a stale checkpoint from a previous run
        # rolling_counter = np.zeros(50) # Maybe we need???
        
        for episode in range(1000000):

            learning_rate = 0.0001
            V_lrn = 8  # Critic learning rate multiplier

            random_angle = np.clip(np.random.normal(0, np.pi/30), -np.pi/15, np.pi/15)
            random_location = np.clip(np.random.normal(2, 1), 0, 4)

            learning_rate_discount = 1

            total_episode_reward = 0

            # Shared buffer across both sides so each batch mixes both starting configs.
            # Randomized order removes the systematic bias of always training side -1 first.
            states_memory = []
            target_V_memory = []
            target_mu_memory = []
            self.d_log_std = 0
            runtimes = []
            episode_advantages = []
            sides = [-1, 1]

            for side in sides:
                self.model.state = [side * random_location, np.pi + side * random_angle, 0, 0]
                t = 0
                done = False
                while not done:
                    t += 1

                    # nn[0] = V (Score), nn[1] = mu (Action)
                    # Normalize before asking for an action
                    normalized_state = self.normalize(self.model.state)

                    V = self.NN_V.feedforward(normalized_state)[-1][0][0]
                    mu = self.NN_mu.feedforward(normalized_state)[-1][0][0]
                    self.model.motor_force = (mu - 0.5) * 200


                    # --- THE PHYSICS ENGINE ---
                    # The cart moves for 0.02 seconds using the chosen force
                    next_state = self.model.rk4_step()
                    reward = self.reward(next_state)
                    total_episode_reward += reward
                    # has_nan = np.isnan(next_state).any()
                    done = next_state[1] <= np.pi/2 or next_state[1] >= 3*np.pi/2 or t >= (max_runtime) * self.model.refresh_rate or abs(next_state[2]) > 100 or abs(next_state[3]) > 100 or abs(next_state[0]) > 100

                    normalized_next_state = self.normalize(next_state)
                    next_critic = self.NN_V.feedforward(normalized_next_state)[-1][0][0]

                    if done and t < (max_runtime) * self.model.refresh_rate:
                        target_value = reward                             # true terminal: no future
                    else:
                        target_value = reward + gamma * next_critic       # alive, or truncated at timeout
                    # Advantage: Was the move better than the Critic expected?
                    advantage_unclipped = target_value - V
                    episode_advantages.append(advantage_unclipped)
                    advantage = np.clip(advantage_unclipped, -5.0, 5.0)

                    # --- THE BACKWARD PASS ---
                    # 1. Backprop for the Critic (Mean Squared Error)
                    # Loss = 0.5 * (target_value - critic)^2
                    # dL/dV = -(target_value - critic) = -advantage
                    d_V = -advantage

                    # Move to the next frame
                    self.model.state = next_state

                    states_memory.append(normalized_state)
                    target_V = V + advantage_unclipped  # Use unclipped advantage for the Critic target to avoid biasing the Critic towards underestimating the value of states 
                    # target_mu = np.clip(target_mu, 0.05, 0.95)
                    target_V = np.clip(target_V, -200, 200.0)

                    target_V_memory.append(target_V)

                    batch_size = 60

                    if len(states_memory) >= batch_size:
                        self.NN_V.backward(np.array(states_memory), np.array(target_V_memory).reshape(-1, 1),  learning_rate * V_lrn / batch_size)

                        states_memory = []
                        target_V_memory = []

                runtimes.append(t)

            # Flush any remaining experience after both sides complete
            if len(states_memory) > 0:
                self.NN_V.backward(np.array(states_memory), np.array(target_V_memory).reshape(-1, 1), learning_rate * V_lrn / batch_size)


            print(f"Episode {episode} finished! Total Reward: {total_episode_reward:.2f}, runtime = {runtimes[0]}, {runtimes[1]}")

            if episode == 0:
                best_episodes.append(episode)
                best_rewards.append(total_episode_reward)
                best_reward = total_episode_reward
                second_best_reward = total_episode_reward
            
            elif total_episode_reward >= best_reward and total_episode_reward <= best_reward * 2 or episode == 0:  # Only consider it a new best if it's not an outlier that might be a lucky fluke
                second_best_reward = best_reward
                best_reward = total_episode_reward
                best_episodes.append(episode)
                best_rewards.append(total_episode_reward)
                self.NN_V.theta_backup()  # slot0 (old best) -> slot1 before overwriting
                self.NN_V.theta_save()    # current weights -> slot0
                previously_saved = True
                print('Saved to 0!')

            elif total_episode_reward >= second_best_reward and total_episode_reward < best_reward * 2:  # Only update second best if it's not an outlier that might be a lucky fluke
                second_best_reward = total_episode_reward
                self.NN_V.theta_save()    # save without touching slot1 backup
                previously_saved = True
                print('Saved to 0 (Second best)!')
            
            if total_episode_reward > best_reward * 2:  # If we do way better than our current best, it's probably a lucky fluke, so don't update our best or second best records, but do save the weights in case it's a sign of something good to come and we want to be able to recover it if we crash before we see more good episodes
                self.NN_V.theta_save(3)  # save to slot3 for recovery, not policy collapse
                total_episode_reward = best_reward  # Don't let the graphs get messed up by a lucky fluke outlier, but do save the weights in case it's a sign of something good to come and we want to be able to recover it if we crash before we see more good episodes
                print('Saved to 3!')
            
            if episode % 100 == 0 and fail_count == 0:  # Only save periodically if we're not in a failure streak
                self.NN_V.theta_save(2)  # periodic save to slot2 for recovery from crashes, not policy collapse
                print('Periodic Save to 2!')

            if total_episode_reward < max(35, second_best_reward * 0.1) and False:  # If we do very poorly, it's a sign of potential policy collapse, but we only want to trigger on a string of bad luck if we haven't had any recent successes to reassure us that the policy is still viable
                fail_count += 1
            else:
                fail_count = 0

            if fail_count >= 500 and previously_saved: 
                self.NN_mu.theta_recover(i = 0)
                self.NN_V.theta_recover(i = 0)
                self.sync_target(hard=True)
                print('policy_collapse')
                fail_count = 0

            full_frames = max_runtime * self.model.refresh_rate
            full_sides_history.append(sum(1 for r in runtimes if r >= full_frames))
            reward_history.append(total_episode_reward)
            advantage_history.append(np.mean(np.abs(episode_advantages)) if episode_advantages else 0.0)
            signed_advantage_history.append(np.mean(episode_advantages) if episode_advantages else 0.0)

            recent = np.mean(reward_history[-50:])
            # learning_rate_discount = float(np.clip(3 - 3 * recent / 7200, 0.1, 1.0))

            #region Live plot update
            # Redraw the diagnostics with this episode's data appended
            episodes = range(len(reward_history))
            line1.set_data(episodes, reward_history)
            for marker, sides_done in ((marker_both, 2), (marker_one, 1), (marker_neither, 0)):
                marker.set_data([e for e, c in zip(episodes, full_sides_history) if c == sides_done],
                                [r for r, c in zip(reward_history, full_sides_history) if c == sides_done])
            line3.set_data(episodes, advantage_history)
            line4.set_data(episodes, signed_advantage_history)
            marker_best.set_data(best_episodes, best_rewards)
            marker_eval.set_data(eval_episodes, eval_rewards)
            # Step-style running best: hold each record until the next one replaces it
            line_best.set_data(list(best_episodes) + [len(reward_history) - 1],
                               list(best_rewards) + [best_rewards[-1] if best_rewards else 0])
            ax1.relim()
            ax1.autoscale_view()
            ax3.relim()
            ax3.autoscale_view()
            fig.canvas.flush_events()
            #endregion 


        self.NN_mu.theta_save(2)
        self.NN_V.theta_save(2)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    y0 = [0, np.pi, 0, 0]
    SP = physics.SinglePendulum(params=(9.8, 1, 1, 1), y0 = y0, refresh_rate=60)
    main = RL_trainer(SP)

    main.train()