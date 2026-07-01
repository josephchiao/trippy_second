from main_legacy import DoublePendulum
import pid
import numpy as np
import neural_network as nn
import math
from RL_training import RL_trainer
# from main import SinglePendulum

class SP_Controller:
    def __init__(self, pendulum, max_motor_force = 100, target = 0):
        
        self.pendulum = pendulum
        self.motor_force = 0
        self.solution = []
        self.max_motor_force = max_motor_force
        self.target = target

        self.stable_counter = 0
        self.taming_time = 0

        self.position_controller = pid.pid_controller(self.target, self.pendulum.state[0], 4, 0.001, 400)

        self.angular_controller_1 = pid.pid_controller(np.pi, self.pendulum.state[1], 50, 0, 1600)
        self.position_controller_1 = pid.pid_controller(self.target, self.pendulum.state[0], 13, 0, 1800, display = False)

    def solve_step_null_input(self):

        self.motor_force = 0
            
        return self.motor_force
    
    def solve_step_inverted_rod(self, kp = 50, ki = 0, kd = 1800, mode = 'analog'):
        
        '''Rod 1 stablize at upright position, rod 2 stablize at down position'''
            
        offset = 0

        if mode == 'analog':
            
            self.angular_controller_1.location = self.pendulum.state[1]
            self.position_controller_1.location = self.pendulum.state[0]

            # Stage 0: If in excessive motion, stablalize
            if abs(self.pendulum.state[3]) > 40 or abs(self.pendulum.state[4]) > 30 or self.stable_counter == -1:
                state_string = 'excessive motion'
                
                self.motor_force = self.position_controller_1.update()
                self.angular_controller_1.update()

                self.stable_counter = -1
                if abs(self.pendulum.state[4]) < np.pi/2 and abs(self.pendulum.state[3]) < 2:
                    self.stable_counter = 0

            # Stage 1: Initialize swing
            elif abs(self.pendulum.state[1]) < 0.01 and abs(self.pendulum.state[4]) < 0.01:
                state_string = 'initialize swing'
                
                self.motor_force = 10

                self.position_controller_1.update()
                self.angular_controller_1.update()

                self.stable_counter = 0
            
            # Stage 2: increase amplitude
            elif self.pendulum.state[1] <= np.pi/2 or self.pendulum.state[1] >= 3 * np.pi/2:
                state_string = 'increase amplitude'

                if self.pendulum.state[4] > 0:
                    self.motor_force = -self.max_motor_force * math.cos(self.pendulum.state[1])
                else:
                    self.motor_force = self.max_motor_force * math.cos(self.pendulum.state[1])
                self.motor_force *= 0.16

                self.position_controller_1.update()
                self.angular_controller_1.update()

                self.stable_counter = 0
            
            # Stage 3: Kick to inverted position
            elif self.pendulum.state[1] >= np.pi/2 and self.pendulum.state[1] <= 3 * np.pi/2 and abs(self.pendulum.state[1] - np.pi + 0.2 * math.atan(offset)) >= np.pi/5 and self.stable_counter == 0:
                state_string = 'kick to inverted position'
                self.angular_controller_1.kp = 20
                self.angular_controller_1.kd = 20
                self.angular_controller_1.target = np.pi

                self.motor_force = self.angular_controller_1.update() * 10
                self.position_controller_1.update()
                
                self.stable_counter = 0
                
            # Stage 4: Tame and maintain
            else:
                state_string = 'terminal'
                if self.stable_counter != -2:
                    self.taming_time = 0
                self.taming_time += 1
                self.stable_counter = -2
                
                self.angular_controller_1.kp = 400 * (((-np.cos(self.pendulum.state[2]) + 1) * 6 + 0.22 * (self.pendulum.state[5]**2)) / 12 + 0.08)
                self.angular_controller_1.kd = 1700 * (((-np.cos(self.pendulum.state[2]) + 1) * 6 + 1.5 * (self.pendulum.state[5]**2)) / 12 * 0.1 + 0.9)
                
                offset = -0.1 * self.position_controller_1.update()
                theoratical_target = np.pi + 0.06 * math.atan(offset)

                self.angular_controller_1.target = theoratical_target
                self.motor_force = self.angular_controller_1.update()

                if (self.pendulum.state[2] <= np.pi/15 or self.pendulum.state[2] >= 29 * np.pi/15) and abs(self.pendulum.state[5]) <= np.pi/15 and abs(self.pendulum.state[0] - self.target) <= 0.7:
                    self.stable_counter = 0      
        
        elif mode == 'ML':
            NN = nn.NeuralNetwork((4, 64, 64, 2), [nn.ReLU, nn.ReLU, [nn.linear, nn.sigmoid]], 'nn_library')
            NN.theta_recover(2)
            self.motor_force = (NN.feedforward(RL_trainer.normalize(self, state = self.pendulum.state))[-1][0][1] - 0.5) * 200
            state_string = 'ML control'

        # reject if the motor is asked to do more than it could
        if abs(self.motor_force) >= self.max_motor_force:
            if self.motor_force > 0:
                self.motor_force = self.max_motor_force
            else:
                self.motor_force = -self.max_motor_force
        
        cost = (self.target - self.pendulum.state[0])**2

        return self.motor_force, state_string, cost

class DP_Controller:
    def __init__(self, pendulum, max_motor_force = 100, target = 0):
        
        self.pendulum = pendulum
        self.motor_force = 0
        self.solution = []
        self.max_motor_force = max_motor_force
        self.target = target

        self.stable_counter = 0
        self.taming_time = 0

        self.position_controller = pid.pid_controller(self.target, self.pendulum.state[0], 4, 0.001, 400)

        self.angular_controller_1 = pid.pid_controller(np.pi, self.pendulum.state[1], 50, 0, 1800)
        self.position_controller_1 = pid.pid_controller(self.target, self.pendulum.state[0], 13, 0, 1300, display = False)

        self.angular_controller_2 = pid.pid_controller(np.pi, self.pendulum.state[1], 15, 0.1, 2000)
        self.position_controller_2 = pid.pid_controller(self.target, self.pendulum.state[0], 6, 0, 1000, display = False)
        self.angular2_countroller_2 = pid.pid_controller(np.pi, self.pendulum.state[2], 15, 0.1, 2000)


    def solve_step_null_input(self):

        self.motor_force = 0
            
        return self.motor_force

    def solve_step_stablize_position(self):

        self.position_controller.location = self.pendulum.state[0]
        self.motor_force = min((self.position_controller.update(), self.max_motor_force), key=abs)
    
        cost = (self.target - self.pendulum.state[0])**2

        return self.motor_force, 'Maintain Position', cost

    def solve_step_inverted_rod_1(self, kp = 50, ki = 0, kd = 1800, mode = 'analog'):
        
        '''Rod 1 stablize at upright position, rod 2 stablize at down position'''
            
        offset = 0

        if mode == 'analog':
            
            self.angular_controller_1.location = self.pendulum.state[1]
            self.position_controller_1.location = self.pendulum.state[0]

            # Stage 0: If in excessive motion, stablalize
            if abs(self.pendulum.state[3]) > 40 or abs(self.pendulum.state[4]) > 30 or self.stable_counter == -1:
                state_string = 'excessive motion'
                
                self.motor_force = self.position_controller_1.update()
                self.angular_controller_1.update()

                self.stable_counter = -1
                if abs(self.pendulum.state[4]) < np.pi/2 and abs(self.pendulum.state[3]) < 2:
                    self.stable_counter = 0

            # Stage 1: Initialize swing
            elif abs(self.pendulum.state[1]) < 0.01 and abs(self.pendulum.state[4]) < 0.01:
                state_string = 'initialize swing'
                
                self.motor_force = 10

                self.position_controller_1.update()
                self.angular_controller_1.update()

                self.stable_counter = 0
            
            # Stage 2: increase amplitude
            elif self.pendulum.state[1] <= np.pi/2 or self.pendulum.state[1] >= 3 * np.pi/2:
                state_string = 'increase amplitude'

                if self.pendulum.state[4] > 0:
                    self.motor_force = -self.max_motor_force * math.cos(self.pendulum.state[1])
                else:
                    self.motor_force = self.max_motor_force * math.cos(self.pendulum.state[1])
                self.motor_force *= 0.16

                self.position_controller_1.update()
                self.angular_controller_1.update()

                self.stable_counter = 0
            
            # Stage 3: Kick to inverted position
            elif self.pendulum.state[1] >= np.pi/2 and self.pendulum.state[1] <= 3 * np.pi/2 and abs(self.pendulum.state[1] - np.pi + 0.2 * math.atan(offset)) >= np.pi/5 and self.stable_counter == 0:
                state_string = 'kick to inverted position'
                self.angular_controller_1.kp = 20
                self.angular_controller_1.kd = 20
                self.angular_controller_1.target = np.pi

                self.motor_force = self.angular_controller_1.update() * 10
                self.position_controller_1.update()
                
                self.stable_counter = 0
                
            # Stage 4: Tame and maintain
            else:
                state_string = 'terminal'
                if self.stable_counter != -2:
                    self.taming_time = 0
                self.taming_time += 1
                self.stable_counter = -2
                
                self.angular_controller_1.kp = 400 * (((-np.cos(self.pendulum.state[2]) + 1) * 6 + 0.22 * (self.pendulum.state[5]**2)) / 12 + 0.08)
                self.angular_controller_1.kd = 1700 * (((-np.cos(self.pendulum.state[2]) + 1) * 6 + 1.5 * (self.pendulum.state[5]**2)) / 12 * 0.1 + 0.9)
                
                offset = -0.1 * self.position_controller_1.update()
                theoratical_target = np.pi + 0.06 * math.atan(offset)

                self.angular_controller_1.target = theoratical_target
                self.motor_force = self.angular_controller_1.update()

                if (self.pendulum.state[2] <= np.pi/15 or self.pendulum.state[2] >= 29 * np.pi/15) and abs(self.pendulum.state[5]) <= np.pi/15 and abs(self.pendulum.state[0] - self.target) <= 0.7:
                    self.stable_counter = 0      
        
        # reject if the motor is asked to do more than it could
        if abs(self.motor_force) >= self.max_motor_force:
            if self.motor_force > 0:
                self.motor_force = self.max_motor_force
            else:
                self.motor_force = -self.max_motor_force
        
        cost = (self.target - self.pendulum.state[0])**2

        return self.motor_force, state_string, cost

    def solve_step_inverted_rod_2(self, kp = 15, ki = 0.1, kd = 2000, mode = 'analog'):
        
        '''Both rods stablize at upright position'''
        
            
        offset = 0

        if mode == 'analog':
            
            self.angular_controller_2.location = self.pendulum.state[1]
            self.position_controller_2.location = self.pendulum.state[0]
            self.angular2_countroller_2.location = self.pendulum.state[2]
            # state_energy = ((-np.cos(self.pendulum.state[2]) + 1) * 6 + 0.42 * (self.pendulum.state[5]**2)) / 12  # 0 when stationary at bottom, 1 when stationary at top

            # Stage 0: If in excessive motion, stablalize
            if abs(self.pendulum.state[3]) > 40 or abs(self.pendulum.state[4]) > 30 or abs(self.pendulum.state[5]) > 30 or self.stable_counter == -1:
                state_string = 'excessive motion'
                
                self.motor_force = self.position_controller_2.update()
                self.angular_controller_2.update()
                self.angular2_countroller_2.update()

                self.stable_counter = -1
                if abs(self.pendulum.state[4]) < np.pi/2 and abs(self.pendulum.state[3]) < 2 and abs(self.pendulum.state[5]) < np.pi/2:
                    self.stable_counter = 0

            # Stage 1: Initialize swing
            elif abs(self.pendulum.state[1]) < 0.01 and abs(self.pendulum.state[4]) < 0.01:
                state_string = 'initialize swing'
                
                self.motor_force = 10

                self.position_controller_2.update()
                self.angular_controller_2.update()
                self.angular2_countroller_2.update()

                self.stable_counter = 0
            
            # Stage 2: increase amplitude
            elif self.pendulum.state[1] <= np.pi/2 or self.pendulum.state[1] >= 3 * np.pi/2:
                state_string = 'increase amplitude'

                if self.pendulum.state[4] > 0:
                    self.motor_force = -self.max_motor_force * math.cos(self.pendulum.state[1])
                else:
                    self.motor_force = self.max_motor_force * math.cos(self.pendulum.state[1])
                self.motor_force *= 0.16

                self.position_controller_2.update()
                self.angular_controller_2.update()
                self.angular2_countroller_2.update()

                self.stable_counter = 0
            
            # Stage 3: Kick to inverted position
            elif self.pendulum.state[1] >= np.pi/2 and self.pendulum.state[1] <= 3 * np.pi/2 and abs(self.pendulum.state[1] - np.pi + 0.2 * math.atan(offset)) >= np.pi/5 and self.stable_counter == 0:
                state_string = 'kick rod1 to inverted position'
                self.angular_controller_2.kp = 20
                self.angular_controller_2.kd = 20
                self.angular_controller_2.target = np.pi

                self.motor_force = self.angular_controller_2.update() * 10
                self.position_controller_2.update()
                self.angular2_countroller_2.update()
                
                self.stable_counter = 0
            
            # Stage 3.5: Kick rod2 to inverted position
            elif (self.pendulum.state[2] >= np.pi/6 or self.pendulum.state[2] <= 11 * np.pi/6 or abs(self.pendulum.state[5]) >= np.pi/3 or self.stable_counter == -2) and self.stable_counter != 1:
                state_string = 'kick rod2 to inverted position'
                if self.stable_counter != -2:
                    self.taming_time = 0
                self.taming_time += 1
                self.stable_counter = -2
                self.angular_controller_2.kp = 260 * ((2 - ((-np.cos(self.pendulum.state[2]) + 1) * 6 + 0.42 * (self.pendulum.state[5]**2)) / 12 ) * 0.4 + 0.4)
                self.angular_controller_2.kd = 700 
                self.angular_controller_2.target = np.pi + self.pendulum.state[5] * 0.0081
                
                self.motor_force = self.angular_controller_2.update()
                self.position_controller_2.update()
                self.angular2_countroller_2.update()

                if self.pendulum.state[2] >= 39 * np.pi/40 and self.pendulum.state[2] <= 41 * np.pi/40:
                    self.stable_counter = 1

            # Stage 4: Maintain 
            else:

                state_string = 'maintain'
                self.angular_controller_2.kp = 1000
                self.angular_controller_2.kd = 1000
                self.angular_controller_2.target = np.pi
                
                offset = self.position_controller_2.update()
                rod2_target = np.pi + min(0.07, (self.pendulum.state[0] - self.target) ** 2 * 0.02) * math.atan(-0.1 * offset)
                rod1_target = 1.21 * (self.pendulum.state[2] - rod2_target) + 0.05 * self.pendulum.state[5] + rod2_target

                self.angular_controller_2.target = rod1_target
                
                self.angular2_countroller_2.update()
                self.motor_force = self.angular_controller_2.update()
                if self.pendulum.state[2] <= 7 * np.pi/8 or self.pendulum.state[2] >= 9 * np.pi/8:
                    self.stable_counter = -2
                            
        # reject if the motor is asked to do more than it could
        if abs(self.motor_force) >= self.max_motor_force:
            if self.motor_force > 0:
                self.motor_force = self.max_motor_force
            else:
                self.motor_force = -self.max_motor_force

        cost = (self.target - self.pendulum.state[0])**2
            
        return self.motor_force, state_string, cost