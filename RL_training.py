import neural_network as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
import random
import physics

init_log_std = -0.5  # Initial exploration noise level (log scale)

class RL_trainer:

    def __init__(self, model):
        
        self.model = model
        self.log_std = init_log_std
        self.log_floor = -2
        self.log_ceiling = 3
        self.d_log_std = 0
        self.NN_V = nn.NeuralNetwork((4, 16, 16, 1), [nn.ReLU, nn.ReLU, nn.linear], 'V_nn_library')
        self.NN_mu = nn.NeuralNetwork((4, 16, 16, 1), [nn.ReLU, nn.ReLU, nn.sigmoid], 'mu_nn_library')

        self.NN_V.theta_generate()
        self.NN_mu.theta_generate()

        self.NN_V.theta_recover()
        self.NN_mu.theta_recover()

        self.NN_V_tgt = nn.NeuralNetwork((4, 16, 16, 1), [nn.ReLU, nn.ReLU, nn.linear], 'V_nn_library')
        self.sync_target(hard=True)

    def reward(self, state):

        """Max reward should be 1. Reward is based on how upright the pendulum is and how close the cart is to the center."""

        location_cf = 0
        angle_cf = 0.8
        time_reward = 0.2
    
        reward = (time_reward - angle_cf * math.cos(state[1]) + location_cf/(4*abs(state[0])+1))

        return reward

    def normalize(self, state):
        """Normalize terms for the state to be readable by the neural network."""
        return np.array([2/(1 + np.exp(-2 * state[0])) - 1, state[1], state[2], state[3]])


    def sync_target(self, hard=False, tau=0.05):
            """Blend the live critic into the frozen target. hard=True copies outright."""
            if hard:
                self.NN_V_tgt.theta = [w.copy() for w in self.NN_V.theta]
                self.NN_V_tgt.b     = [w.copy() for w in self.NN_V.b]
            else:
                for i in range(len(self.NN_V.theta)):
                    self.NN_V_tgt.theta[i] = (1 - tau) * self.NN_V_tgt.theta[i] + tau * self.NN_V.theta[i]
                    self.NN_V_tgt.b[i]     = (1 - tau) * self.NN_V_tgt.b[i]     + tau * self.NN_V.b[i]

    def backward_std(self, action, mu, sigma, advantage):
        epsilon = 1e-12  # Small constant to prevent division by zero
        action_discrepency = action / 200 + 0.5 - mu

        d_mu = -advantage * (action_discrepency / (sigma ** 2 + epsilon))
        # d_mu = -advantage * action_discrepency
        # 1. Calculate the gradient for this specific frame
        step_d_log_std = -advantage * ((action_discrepency**2 / (sigma ** 2 + epsilon)) - 1.0)
        
        # if abs(step_d_log_std) > 20.0: 
        #     step_d_log_std = np.sign(step_d_log_std) * 20.0
            
        self.d_log_std += step_d_log_std
        
        return d_mu   
    
    def train(self, variance = 0, max_runtime = 60):

        t = 0
        solution = []
        state_history = []
        total_cost = 0


        gamma = 0.99  # Discount factor (how much we care about the future)
        reward_history = []
        log_std_history = []
        fail_count = 0

        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Log Std')
        line1, = ax1.plot([], [])
        line2, = ax2.plot([], [])
        best_reward = 0
        second_best_reward = 0
        previously_saved = False  # don't let recovery load a stale checkpoint from a previous run
        # rolling_counter = np.zeros(50) # Maybe we need???

        for episode in range(1000000):

            learning_rate = 0.00003
            V_lrn = 5  # Critic learning rate multiplier
            random_angle = np.pi/120
            random_location = 0

            total_episode_reward = 0

            # Shared buffer across both sides so each batch mixes both starting configs.
            # Randomized order removes the systematic bias of always training side -1 first.
            states_memory = []
            target_V_memory = []
            target_mu_memory = []
            self.d_log_std = 0
            runtimes = []
            sides = [-1, 1]
            random.shuffle(sides)

            for side in sides:
                self.model.state = [0, np.pi + side * random_angle, 0, 0]
                t = 0
                done = False
                while not done:
                    t += 1

                    # nn[0] = V (Score), nn[1] = mu (Action)
                    # Normalize before asking for an action
                    normalized_state = self.normalize(self.model.state)

                    V = self.NN_V_tgt.feedforward(normalized_state)[-1][0][0]
                    mu = self.NN_mu.feedforward(normalized_state)[-1][0][0]
                    self.model.motor_force = (mu - 0.5) * 200 + np.exp(self.log_std) * np.random.randn()


                    # --- THE PHYSICS ENGINE ---
                    # The cart moves for 0.02 seconds using the chosen force
                    next_state = self.model.rk4_step()
                    reward = self.reward(next_state)
                    total_episode_reward += reward
                    # has_nan = np.isnan(next_state).any()
                    done = next_state[1] <= np.pi/2 or next_state[1] >= 3*np.pi/2 or t >= (max_runtime) * self.model.refresh_rate or abs(next_state[2]) > 100 or abs(next_state[3]) > 100

                    normalized_next_state = self.normalize(next_state)
                    next_critic = self.NN_V.feedforward(normalized_next_state)[-1][0][0]

                    if done and t < (max_runtime) * self.model.refresh_rate:
                        target_value = reward                             # true terminal: no future
                    else:
                        target_value = reward + gamma * next_critic       # alive, or truncated at timeout
                    # Advantage: Was the move better than the Critic expected?
                    advantage_unclipped = target_value - V
                    advantage = np.clip(advantage_unclipped, -5.0, 5.0)

                    # --- THE BACKWARD PASS ---
                    # 1. Backprop for the Critic (Mean Squared Error)
                    # Loss = 0.5 * (target_value - critic)^2
                    # dL/dV = -(target_value - critic) = -advantage
                    d_V = -advantage

                    # 2. Backprop for the Actor
                    # This function updates self.d_log_std internally and returns the gradient for mu
                    d_mu = self.backward_std(
                        action=self.model.motor_force,
                        mu=mu,
                        sigma=np.exp(self.log_std)/200,
                        advantage=advantage)

                    # Capping function
                    # if abs(d_V) > 1.0: d_V = np.sign(d_V) * 1.0
                    # if abs(d_mu) > 0.25: d_mu = np.sign(d_mu) * 0.25

                    # Move to the next frame
                    self.model.state = next_state

                    states_memory.append(normalized_state)
                    target_V = V + advantage_unclipped  # Use unclipped advantage for the Critic target to avoid biasing the Critic towards underestimating the value of states 
                    target_mu = mu - d_mu
                    # target_mu = np.clip(target_mu, 0.05, 0.95)

                    target_V_memory.append(target_V)
                    target_mu_memory.append(target_mu)

                    batch_size = 32

                    if len(states_memory) >= batch_size:
                        self.NN_mu.backward(np.array(states_memory), np.array(target_mu_memory).reshape(-1, 1), learning_rate / batch_size)
                        self.NN_V.backward(np.array(states_memory), np.array(target_V_memory).reshape(-1, 1), learning_rate * V_lrn / batch_size)

                        # Update exploration noise (entropy_coeff resists collapse to floor)
                        entropy_coeff = 0.05
                        self.log_std -= learning_rate * (self.d_log_std / batch_size - entropy_coeff)
                        self.log_std = np.clip(self.log_std, self.log_floor, self.log_ceiling)

                        states_memory = []
                        target_V_memory = []
                        target_mu_memory = []
                        self.d_log_std = 0

                runtimes.append(t)

            # Flush any remaining experience after both sides complete
            if len(states_memory) > 0:
                entropy_coeff = 0.05
                self.NN_mu.backward(np.array(states_memory), np.array(target_mu_memory).reshape(-1, 1), learning_rate / batch_size)
                self.NN_V.backward(np.array(states_memory), np.array(target_V_memory).reshape(-1, 1), learning_rate * V_lrn / batch_size)
                self.log_std -= learning_rate * (self.d_log_std / len(states_memory) - entropy_coeff)
                self.log_std = np.clip(self.log_std, self.log_floor, self.log_ceiling)

            print(f"Episode {episode} finished! Total Reward: {total_episode_reward:.2f}, runtime = {runtimes[0]}, {runtimes[1]}")

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
            
            if total_episode_reward > best_reward * 2:  # If we do way better than our current best, it's probably a lucky fluke, so don't update our best or second best records, but do save the weights in case it's a sign of something good to come and we want to be able to recover it if we crash before we see more good episodes
                self.NN_mu.theta_save(3)  # save to slot3 for recovery, not policy collapse
                self.NN_V.theta_save(3)  # save to slot3 for recovery, not policy collapse
                total_episode_reward = best_reward  # Don't let the graphs get messed up by a lucky fluke outlier, but do save the weights in case it's a sign of something good to come and we want to be able to recover it if we crash before we see more good episodes
                print('Saved to 3!')
            
            if episode % 100 == 0 and fail_count == 0:  # Only save periodically if we're not in a failure streak
                self.NN_mu.theta_save(2)  # periodic save to slot2 for recovery from crashes, not policy collapse
                self.NN_V.theta_save(2)  # periodic save to slot2 for recovery from crashes, not policy collapse
                print('Periodic Save to 2!')

            if total_episode_reward < max(60, second_best_reward * 0.05):  # If we do very poorly, it's a sign of potential policy collapse, but we only want to trigger on a string of bad luck if we haven't had any recent successes to reassure us that the policy is still viable
                fail_count += 1
            else:
                fail_count = 0

            if fail_count >= 500 and previously_saved:
                self.NN_mu.theta_recover(i = 1)
                self.NN_V.theta_recover(i = 1)
                self.sync_target(hard=True)
                print('policy_collapse')
                self.log_std = init_log_std  # restore exploration, not kill it
                fail_count = 0

            self.sync_target(tau=0.05)
            reward_history.append(total_episode_reward)
            log_std_history.append(self.log_std)

            episodes = range(len(reward_history))
            line1.set_data(episodes, reward_history)
            line2.set_data(episodes, log_std_history)
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            fig.canvas.flush_events()

        self.NN_mu.theta_save(2)
        self.NN_V.theta_save(2)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    y0 = [0, np.pi, 0, 0]
    SP = physics.SinglePendulum(params=(9.8, 1, 1, 1), y0 = y0, refresh_rate=60)
    main = RL_trainer(SP)

    variance = 1
    main.train(variance = variance)
