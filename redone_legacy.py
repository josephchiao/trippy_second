import math
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import sympy as sm
import pid 
import neural_network as nn

class DoublePendulum:
    """
    A class representing a Double Pendulum on a cart system.
    This class encapsulates the physics engine and numerical integration
    to simulate the dynamics of the system.

    angle = 0 at straight down.
    """
    def __init__(self, params = (9.81, 1, 1, 1, 1, 1), y0 = [0, 0, 0, 0, 0, 0], t_start = 0.0, t_end = 20, fps = 60, max_motor_force = 100, target = 0):
        self.g = 9.81
        self.calc_M, self.calc_F = self.physics_engine()
        self.params = params  # gravity, mass of cart, mass rod1, mass rod2, length rod1, length rod2
        self.state = y0
        self.motor_force = 0
        self.t_start = t_start
        self.t_end = t_end      # Simulate 10 seconds
        self.fps = fps          # Animation frames per second
        self.current_time = t_start
        self.t_eval = np.linspace(t_start, t_end, int((t_end - t_start) * fps))
        self.dt = self.t_eval[1] - self.t_eval[0]
        self.solution = []
        self.max_motor_force = max_motor_force
        self.target = target

    def physics_engine(self):
        # 1. Define Time and Constants
        t = sm.Symbol('t')
        g = sm.Symbol('g')
        m_c, m1, m2 = sm.symbols('m_c m1 m2')  # Masses
        L1, L2 = sm.symbols('L1 L2')           # Lengths
        I1, I2 = sm.symbols('I1 I2')           # Moments of Inertia (e.g., 1/12 * m * L**2)

        # 2. Define Generalized Coordinates (Functions of time)
        x = sm.Function('x')(t)
        th1 = sm.Function('th1')(t)
        th2 = sm.Function('th2')(t)

        q = [x, th1, th2]
        dq = [sm.diff(qi, t) for qi in q]
        ddq = [sm.diff(dqi, t) for dqi in dq]

        # Extract velocities for cleaner math below
        dx, dth1, dth2 = dq

        # 3. Define Center of Mass (CoM) Positions
        # Cart
        x_c = x
        y_c = 0

        # Rod 1 (CoM is halfway down L1)
        x1 = x + (L1 / 2) * sm.sin(th1)
        y1 = -(L1 / 2) * sm.cos(th1)

        # Rod 2 (Attached to end of L1, CoM is halfway down L2)
        x2 = x + L1 * sm.sin(th1) + (L2 / 2) * sm.sin(th2)
        y2 = -L1 * sm.cos(th1) - (L2 / 2) * sm.cos(th2)

        # 4. Calculate Velocities (Time derivatives of positions)
        v_c_x = sm.diff(x_c, t)

        v1_x = sm.diff(x1, t)
        v1_y = sm.diff(y1, t)

        v2_x = sm.diff(x2, t)
        v2_y = sm.diff(y2, t)

        # 5. Define Energies
        # Kinetic Energy: T = T_trans_cart + (T_trans_1 + T_rot_1) + (T_trans_2 + T_rot_2)
        T_cart = 0.5 * m_c * v_c_x**2
        T1 = 0.5 * m1 * (v1_x**2 + v1_y**2) + 0.5 * I1 * dth1**2
        T2 = 0.5 * m2 * (v2_x**2 + v2_y**2) + 0.5 * I2 * dth2**2
        T = T_cart + T1 + T2

        # Potential Energy: V = m * g * y_com
        V = m1 * g * y1 + m2 * g * y2

        # 6. Build the Lagrangian
        L = T - V

        # 7. Apply Euler-Lagrange
        equations = []
        for i, qi in enumerate(q):
            dL_ddq = sm.diff(L, dq[i])
            term1 = sm.diff(dL_ddq, t)
            term2 = sm.diff(L, qi)
            equations.append(term1 - term2)

        # 8. Extract the Mass Matrix (M) and the Forcing Vector (F)
        eq_matrix = sm.Matrix(equations)

        # The Jacobian of the equations with respect to the accelerations IS the Mass matrix
        M = eq_matrix.jacobian(ddq)

        # The rest of the terms (Coriolis, Gravity) are found by setting accelerations to 0
        F = -eq_matrix.subs({ddq_i: 0 for ddq_i in ddq})

        # 9. Compile M and F into fast NumPy functions
        # Notice we no longer need dx, dth1, dth2 for the Mass matrix
        inputs = [g, m_c, m1, m2, L1, L2, I1, I2, x, th1, th2, dx, dth1, dth2]
        M_inputs = [m_c, m1, m2, L1, L2, I1, I2, th1, th2] # M only depends on positions and constants

        # These will compile almost instantly
        calc_M = sm.lambdify(M_inputs, M, "numpy")
        calc_F = sm.lambdify(inputs, F, "numpy")

        print("Ready")
        return calc_M, calc_F

    def get_accelerations(self, state):
        x, th1, th2, dx, dth1, dth2 = state
        g, m_c, m1, m2, L1, L2 = self.params
        I1 = 1/3 * m1 * L1**2
        I2 = 1/3 * m2 * L2**2

        
        # Calculate physics matrices (from SymPy)
        M_num = self.calc_M(m_c, m1, m2, L1, L2, I1, I2, th1, th2)
        F_num = self.calc_F(g, m_c, m1, m2, L1, L2, I1, I2, x, th1, th2, dx, dth1, dth2).flatten()
        
        # Add your motor input! 
        # (Assuming index 0 is the cart's linear x-axis)
        tau = np.array([self.motor_force, 0.0, 0.0])
        
        # Solve for accelerations
        ddq = np.linalg.solve(M_num, F_num + tau)
        
        # Return the first-order derivatives: [velocities, accelerations]
        return np.array([dx, dth1, dth2, ddq[0], ddq[1], ddq[2]])

    def rk4_step(self):
        """Calculates the next state of the system using RK4 integration."""
        
        # k1: Slope at the beginning of the interval
        k1 = self.get_accelerations(self.state)
        
        # k2: Slope at the midpoint (using k1)
        k2 = self.get_accelerations(self.state + 0.5 * self.dt * k1)
        
        # k3: Slope at the midpoint (using k2)
        k3 = self.get_accelerations(self.state + 0.5 * self.dt * k2)
        
        # k4: Slope at the end of the interval (using k3)
        k4 = self.get_accelerations(self.state + self.dt * k3)
        
        # Weighted average of the slopes yields the new state
        new_state = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        while new_state[1] > 2 * np.pi:
            new_state[1] -= 2*np.pi
        while new_state[1] < 0:
            new_state[1] += 2*np.pi
        while new_state[2] > 2 * np.pi:
            new_state[2] -= 2*np.pi
        while new_state[2] < 0:
            new_state[2] += 2*np.pi
        
        return new_state

    def solve_step_null_input(self):

        for t in self.t_eval:

            # 1. Decide on your motor torque for this specific frame
            self.motor_force = 0

            # 2. Advance the physics by exactly one frame (dt)
            self.state = self.rk4_step()
            self.solution.append(np.append(self.state, 0))
            self.current_time += self.dt
            
        return np.array(self.solution), 0

    def solve_step_stablize_position(self, kp = 4, ki = 0.001, kd = 400):

        controller = pid.pid_controller(self.target, self.state[0], kp, ki, kd)
        cost = 0

        for t in self.t_eval:

            # 1. Decide on your motor torque for this specific frame
            controller.location = self.state[0]
            if t >= 2:
                self.motor_force = min((controller.update(), self.max_motor_force), key=abs)
            else:
                self.motor_force = 0
            # 2. Advance the physics by exactly one frame (dt)
            self.state = self.rk4_step()
            self.solution.append(np.append(self.state, self.motor_force))
            self.current_time += self.dt
            cost += (self.target - self.state[0])**2
            
        return np.array(self.solution), cost

    def solve_step_inverted_rod_1(self, kp = 50, ki = 0, kd = 1800, mode = 'analog'):
        
        '''Rod 1 stablize at upright position, rod 2 stablize at down position'''
        
        if mode == 'analog':
            angular_controller = pid.pid_controller(np.pi, self.state[1], kp, ki, kd)
            position_controller = pid.pid_controller(self.target, self.state[0], 13, 0, 1300, display = False)
        elif mode == 'RL':
            NN = nn.NeuralNetwork((6, 64, 64, 2), [nn.ReLU, nn.ReLU, [nn.linear, nn.sigmoid]], 'nn_library')
            NN.theta_recover()
            
        cost = 0
        offset = 0
        stable_counter = 0 
        state_history = []

        for t in self.t_eval:
            if mode == 'analog':
                
                angular_controller.location = self.state[1]
                position_controller.location = self.state[0]

                # Stage 0: If in excessive motion, stablalize
                if abs(self.state[3]) > 40 or abs(self.state[4]) > 30 or stable_counter == -1:
                    state_string = 'excessive motion'
                    
                    self.motor_force = position_controller.update()
                    angular_controller.update()

                    stable_counter = -1
                    if abs(self.state[4]) < np.pi/2 and abs(self.state[3]) < 2:
                        stable_counter = 0

                # Stage 1: Initialize swing
                elif abs(self.state[1]) < 0.01 and abs(self.state[4]) < 0.01:
                    state_string = 'initialize swing'
                    
                    self.motor_force = 10

                    position_controller.update()
                    angular_controller.update()

                    stable_counter = 0
                
                # Stage 2: increase amplitude
                elif self.state[1] <= np.pi/2 or self.state[1] >= 3 * np.pi/2:
                    state_string = 'increase amplitude'

                    if self.state[4] > 0:
                        self.motor_force = -self.max_motor_force * math.cos(self.state[1])
                    else:
                        self.motor_force = self.max_motor_force * math.cos(self.state[1])
                    self.motor_force *= 0.16

                    position_controller.update()
                    angular_controller.update()

                    stable_counter = 0
                
                # Stage 3: Kick to inverted position
                elif self.state[1] >= np.pi/2 and self.state[1] <= 3 * np.pi/2 and abs(self.state[1] - np.pi + 0.2 * math.atan(offset)) >= np.pi/5 and stable_counter == 0:
                    state_string = 'kick to inverted position'
                    angular_controller.kp = 20
                    angular_controller.kd = 20
                    angular_controller.target = np.pi

                    self.motor_force = angular_controller.update() * 10
                    position_controller.update()
                    
                    stable_counter = 0
                    
                # Stage 4: Tame and maintain
                else:
                    state_string = 'terminal'
                    if stable_counter != -2:
                        taming_time = 0
                    taming_time += 1
                    stable_counter = -2
                    angular_controller.kp = 600 * (((-np.cos(self.state[2]) + 1) * 6 + 0.42 * (self.state[5]**2)) / 12 + 0.08)
                    angular_controller.kd = 1400 * (((-np.cos(self.state[2]) + 1) * 6 + 0.42 * (self.state[5]**2)) / 12 * 0.1 + 0.9)
                    offset = -0.1 * position_controller.update()
                    theoratical_target = np.pi + 0.03 * math.atan(offset)

                    # angular_controller.target = np.pi + math.cos(taming_time/60 * 1.1) * 0.1
                    # if self.state[2] <= np.pi:
                    #     angular_controller.target = theoratical_target + self.state[2] * (0.35 / (max(taming_time - 240, 0)/90 + 1) + 0.12)
                    # else:
                    #     angular_controller.target = theoratical_target - (2 * np.pi - self.state[2]) * (0.35 / (max(taming_time - 240, 0)/90 + 1) + 0.12)

                    angular_controller.target = theoratical_target
                    self.motor_force = angular_controller.update()

                    if (self.state[2] <= np.pi/15 or self.state[2] >= 29 * np.pi/15) and abs(self.state[5]) <= np.pi/15 and abs(self.state[0] - self.target) <= 0.7:
                        stable_counter = 0 


                    

            elif mode == 'RL':
                self.motor_force = NN.feedforward(self.state)[-1][0][1] * 100
            
            # reject if the motor is asked to do more than it could
            if abs(self.motor_force) >= self.max_motor_force:
                print('Overload at ', t, 's')
                if self.motor_force > 0:
                    self.motor_force = self.max_motor_force
                else:
                    self.motor_force = -self.max_motor_force

            # 2. Advance the physics by exactly one frame (dt)
            self.state = self.rk4_step()
            self.solution.append(np.append(self.state, self.motor_force))
            state_history.append(state_string)

            self.current_time += self.dt
            cost += (self.target - self.state[0])**2
            
        return np.array(self.solution), cost, state_history

    def solve_step_inverted_rod_2(self, kp = 15, ki = 0.1, kd = 2000, mode = 'analog'):
        
        '''Both rods stablize at upright position'''
        
        if mode == 'analog':
            angular_controller = pid.pid_controller(np.pi, self.state[1], kp, ki, kd)
            position_controller = pid.pid_controller(self.target, self.state[0], 2, 0, 500, display = False)
            angular2_countroller = pid.pid_controller(np.pi, self.state[2], kp, ki, kd)
        elif mode == 'RL':
            NN = nn.NeuralNetwork((6, 64, 64, 2), [nn.ReLU, nn.ReLU, [nn.linear, nn.sigmoid]], 'nn_library')
            NN.theta_recover()
            
        cost = 0
        offset = 0
        stable_counter = 0 
        state_history = []

        for t in self.t_eval:
            if mode == 'analog':
                
                angular_controller.location = self.state[1]
                position_controller.location = self.state[0]
                angular2_countroller.location = self.state[2]
                # state_energy = ((-np.cos(self.state[2]) + 1) * 6 + 0.42 * (self.state[5]**2)) / 12  # 0 when stationary at bottom, 1 when stationary at top

                # Stage 0: If in excessive motion, stablalize
                if abs(self.state[3]) > 40 or abs(self.state[4]) > 30 or abs(self.state[5]) > 30 or stable_counter == -1:
                    state_string = 'excessive motion'
                    
                    self.motor_force = position_controller.update()
                    angular_controller.update()
                    angular2_countroller.update()

                    stable_counter = -1
                    if abs(self.state[4]) < np.pi/2 and abs(self.state[3]) < 2 and abs(self.state[5]) < np.pi/2:
                        stable_counter = 0

                # Stage 1: Initialize swing
                elif abs(self.state[1]) < 0.01 and abs(self.state[4]) < 0.01:
                    state_string = 'initialize swing'
                    
                    self.motor_force = 10

                    position_controller.update()
                    angular_controller.update()
                    angular2_countroller.update()

                    stable_counter = 0
                
                # Stage 2: increase amplitude
                elif self.state[1] <= np.pi/2 or self.state[1] >= 3 * np.pi/2:
                    state_string = 'increase amplitude'

                    if self.state[4] > 0:
                        self.motor_force = -self.max_motor_force * math.cos(self.state[1])
                    else:
                        self.motor_force = self.max_motor_force * math.cos(self.state[1])
                    self.motor_force *= 0.16

                    position_controller.update()
                    angular_controller.update()
                    angular2_countroller.update()

                    stable_counter = 0
                
                # Stage 3: Kick to inverted position
                elif self.state[1] >= np.pi/2 and self.state[1] <= 3 * np.pi/2 and abs(self.state[1] - np.pi + 0.2 * math.atan(offset)) >= np.pi/5 and stable_counter == 0:
                    state_string = 'kick rod1 to inverted position'
                    angular_controller.kp = 20
                    angular_controller.kd = 20
                    angular_controller.target = np.pi

                    self.motor_force = angular_controller.update() * 10
                    position_controller.update()
                    angular2_countroller.update()
                    
                    stable_counter = 0
                
                # Stage 3.5: Kick rod2 to inverted position
                elif (self.state[2] >= np.pi/6 or self.state[2] <= 11 * np.pi/6 or abs(self.state[5]) >= np.pi/3 or stable_counter == -2) and stable_counter != 1:
                    state_string = 'kick rod2 to inverted position'
                    if stable_counter != -2:
                        taming_time = 0
                    taming_time += 1
                    stable_counter = -2
                    angular_controller.kp = 260 * ((2 - ((-np.cos(self.state[2]) + 1) * 6 + 0.42 * (self.state[5]**2)) / 12 ) * 0.4 + 0.4)
                    angular_controller.kd = 700 
                    angular_controller.target = np.pi + self.state[5] * 0.0081
                    
                    self.motor_force = angular_controller.update()
                    position_controller.update()
                    angular2_countroller.update()

                    if self.state[2] >= 39 * np.pi/40 and self.state[2] <= 41 * np.pi/40:
                        stable_counter = 1

                # Stage 4: Maintain 
                else:

                    state_string = 'maintain'
                    angular_controller.kp = 1000
                    angular_controller.kd = 1000
                    angular_controller.target = np.pi
                    
                    offset = position_controller.update()
                    rod2_target = np.pi + 0.03 * math.atan(-0.1 * offset)
                    rod1_target = 1.21 * (self.state[2] - rod2_target) + 0.05 * self.state[5] + rod2_target

                    angular_controller.target = rod1_target
                    
                    angular2_countroller.update()
                    self.motor_force = angular_controller.update()
                    if self.state[2] <= 7 * np.pi/8 or self.state[2] >= 9 * np.pi/8:
                        stable_counter = -2
                    

            elif mode == 'RL':
                self.motor_force = NN.feedforward(self.state)[-1][0][1] * 100
            
            # reject if the motor is asked to do more than it could
            if abs(self.motor_force) >= self.max_motor_force:
                print('Overload at ', t, 's')
                if self.motor_force > 0:
                    self.motor_force = self.max_motor_force
                else:
                    self.motor_force = -self.max_motor_force

            # 2. Advance the physics by exactly one frame (dt)
            self.state = self.rk4_step()
            self.solution.append(np.append(self.state, self.motor_force))
            state_history.append(state_string)

            self.current_time += self.dt
            cost += (self.target - self.state[0])**2
            
        return np.array(self.solution), cost, state_history

    def animate(self, speed = 1):
        solution, cost, state_history = self.solve_step_inverted_rod_2()
        print('cost =', cost)
        # Extract position arrays for the animation
        x_cart_history = solution[:, 0]
        th1_history = solution[:, 1]
        th2_history = solution[:, 2]
        force_history = solution[:, 6]

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-10, 15)
        ax.set_ylim(-3, 8)

        # Setup empty artists
        cart_marker, = ax.plot([], [], 'ks', markersize=10) # Black square for cart
        rod1, = ax.plot([], [], 'b-', lw=2)
        rod2, = ax.plot([], [], 'r-', lw=2)
        rail = ax.plot([-100,100], [0, 0], 'k-', lw=2)

        # --- NEW: Setup the quiver object for the force arrow ---
        # scale_units='xy' and scale=1 means the arrow length matches plot coordinates.
        # We will manually scale the force magnitude below to keep it visually manageable.
        force_arrow = ax.quiver([0], [0], [0], [0], color='green', pivot='tail', 
                                angles='xy', scale_units='xy', scale=1, width=0.01, zorder=5)
        force_scale = 0.1 # Adjust this multiplier to change how long the arrow draws

        # 1. Setup the empty text object for the time
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12)
        location_text = ax.text(0.05, 0.8, '', transform=ax.transAxes, fontsize=12)
        angle_1_text = ax.text(0.05, 0.7, '', transform=ax.transAxes, fontsize=12)
        angle_2_text = ax.text(0.05, 0.6, '', transform=ax.transAxes, fontsize=12)
        force_text = ax.text(0.05, 0.5, '', transform=ax.transAxes, fontsize=12)
        state_text = ax.text(0.05, 0.4, '', transform=ax.transAxes, fontsize=12)

        def init():
            cart_marker.set_data([], [])
            rod1.set_data([], [])
            rod2.set_data([], [])
            
            # Reset the arrow to zero length at the origin
            force_arrow.set_offsets([[0, 0]])
            force_arrow.set_UVC(0, 0)
            
            # 2. Clear the text in the initialization
            time_text.set_text('')
            location_text.set_text('')
            angle_1_text.set_text('')
            angle_2_text.set_text('')
            force_text.set_text('')
            state_text.set_text('')

            # 3. Return the text artist alongside the others, including force_arrow
            return cart_marker, rod1, rod2, force_arrow, time_text, location_text, angle_1_text, angle_2_text, force_text, state_text

        def update(frame):
            x_c = x_cart_history[frame]
            th1 = th1_history[frame]
            th2 = th2_history[frame]
            f = force_history[frame]
            state = state_history[frame]
            
            # --- NEW: Dynamic Camera Tracking ---
            xmin, xmax = ax.get_xlim()
            margin = 4.0 # Pan the camera if the cart gets within 4 units of the edge
            
            if x_c > xmax - margin:
                shift = x_c - (xmax - margin)
                ax.set_xlim(xmin + shift, xmax + shift)
            elif x_c < xmin + margin:
                shift = x_c - (xmin + margin)
                ax.set_xlim(xmin + shift, xmax + shift)
            # ------------------------------------
            
            x1 = x_c + self.params[4] * np.sin(th1)
            y1 = -self.params[4] * np.cos(th1)
            
            x2 = x1 + self.params[5] * np.sin(th2)
            y2 = y1 - self.params[5] * np.cos(th2)
            
            cart_marker.set_data([x_c], [0])
            rod1.set_data([x_c, x1], [0, y1])
            rod2.set_data([x1, x2], [y1, y2])
            
            force_arrow.set_offsets([[x_c, 0]])
            force_arrow.set_UVC(f * force_scale, 0)
            
            current_time = self.t_eval[frame]
            time_text.set_text(f'Time: {current_time:.2f} s')
            location_text.set_text(f'Location: {x_c:.2f} m')
            angle_1_text.set_text(f'Angle 1: {th1:.2f} rad')
            angle_2_text.set_text(f'Angle 2: {th2:.2f} rad')
            force_text.set_text(f'Force: {f:.2f} N')
            state_text.set_text(f'State: {state}')
            
            return cart_marker, rod1, rod2, force_arrow, time_text, location_text, angle_1_text, angle_2_text, force_text, state_text

        ani = animation.FuncAnimation(
            fig, update, frames=len(self.t_eval), 
            init_func=init, 
            blit=False, # <-- CRITICAL: Change blit to False to allow axis updates
            interval=1000/self.fps*speed
        )
        plt.show()

class SinglePendulum:
    def __init__(self, params = (9.81, 1, 1, 1), y0 = [0, np.pi, 0, 0], t_start = 0.0, t_end = 20, fps = 60, max_motor_force = 100, target = 0):
        self.g = 9.81
        self.calc_M, self.calc_F = self.physics_engine()
        self.params = params
        self.state = y0
        self.motor_force = 0
        self.t_start = t_start
        self.t_end = t_end      # Simulate 10 seconds
        self.fps = fps          # Animation frames per second
        self.current_time = t_start
        self.t_eval = np.linspace(t_start, t_end, int((t_end - t_start) * fps))
        self.dt = self.t_eval[1] - self.t_eval[0]
        self.solution = []
        self.max_motor_force = max_motor_force
        self.target = target

    def physics_engine(self):
        # 1. Define Time and Constants
        t = sm.Symbol('t')
        g = sm.Symbol('g')
        m_c, m1 = sm.symbols('m_c m1')  # Masses
        L1 = sm.symbols('L1')           # Lengths
        I1 = sm.symbols('I1')           # Moments of Inertia (e.g., 1/12 * m * L**2)

        # 2. Define Generalized Coordinates (Functions of time)
        x = sm.Function('x')(t)
        th1 = sm.Function('th1')(t)

        q = [x, th1]
        dq = [sm.diff(qi, t) for qi in q]
        ddq = [sm.diff(dqi, t) for dqi in dq]

        # Extract velocities for cleaner math below
        dx, dth1 = dq

        # 3. Define Center of Mass (CoM) Positions
        # Cart
        x_c = x
        y_c = 0

        # Rod 1 (CoM is halfway down L1)
        x1 = x + (L1 / 2) * sm.sin(th1)
        y1 = -(L1 / 2) * sm.cos(th1)

        # 4. Calculate Velocities (Time derivatives of positions)
        v_c_x = sm.diff(x_c, t)

        v1_x = sm.diff(x1, t)
        v1_y = sm.diff(y1, t)

        # 5. Define Energies
        # Kinetic Energy: T = T_trans_cart + (T_trans_1 + T_rot_1) + (T_trans_2 + T_rot_2)
        T_cart = 0.5 * m_c * v_c_x**2
        T1 = 0.5 * m1 * (v1_x**2 + v1_y**2) + 0.5 * I1 * dth1**2
        T = T_cart + T1

        # Potential Energy: V = m * g * y_com
        V = m1 * g * y1

        # 6. Build the Lagrangian
        L = T - V

        # 7. Apply Euler-Lagrange
        equations = []
        for i, qi in enumerate(q):
            dL_ddq = sm.diff(L, dq[i])
            term1 = sm.diff(dL_ddq, t)
            term2 = sm.diff(L, qi)
            equations.append(term1 - term2)

        # 8. Extract the Mass Matrix (M) and the Forcing Vector (F)
        eq_matrix = sm.Matrix(equations)

        # The Jacobian of the equations with respect to the accelerations IS the Mass matrix
        M = eq_matrix.jacobian(ddq)

        # The rest of the terms (Coriolis, Gravity) are found by setting accelerations to 0
        F = -eq_matrix.subs({ddq_i: 0 for ddq_i in ddq})

        # 9. Compile M and F into fast NumPy functions
        # Notice we no longer need dx, dth1, dth2 for the Mass matrix
        inputs = [g, m_c, m1, L1, I1, x, th1, dx, dth1]
        M_inputs = [m_c, m1, L1, I1, th1] # M only depends on positions and constants

        # These will compile almost instantly
        calc_M = sm.lambdify(M_inputs, M, "numpy")
        calc_F = sm.lambdify(inputs, F, "numpy")

        print("Ready")
        return calc_M, calc_F

    def get_accelerations(self, state):
        x, th1, dx, dth1 = state
        g, m_c, m1, L1 = self.params
        I1 = 1/3 * m1 * L1**2

        # Calculate physics matrices (from SymPy)
        M_num = self.calc_M(m_c, m1, L1, I1, th1)
        F_num = self.calc_F(g, m_c, m1, L1, I1, x, th1, dx, dth1).flatten()
        
        # Add your motor input! 
        # (Assuming index 0 is the cart's linear x-axis)
        tau = np.array([self.motor_force, 0.0])
        
        # Solve for accelerations
        ddq = np.linalg.solve(M_num, F_num + tau)
        
        # Return the first-order derivatives: [velocities, accelerations]
        return np.array([dx, dth1, ddq[0], ddq[1]])

    def rk4_step(self):
        """Calculates the next state of the system using RK4 integration."""
        
        # k1: Slope at the beginning of the interval
        k1 = self.get_accelerations(self.state)
        
        # k2: Slope at the midpoint (using k1)
        k2 = self.get_accelerations(self.state + 0.5 * self.dt * k1)
        
        # k3: Slope at the midpoint (using k2)
        k3 = self.get_accelerations(self.state + 0.5 * self.dt * k2)
        
        # k4: Slope at the end of the interval (using k3)
        k4 = self.get_accelerations(self.state + self.dt * k3)
        
        # Weighted average of the slopes yields the new state
        new_state = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        while new_state[1] > 2 * np.pi:
            new_state[1] -= 2*np.pi
        while new_state[1] < 0:
            new_state[1] += 2*np.pi
        
        return new_state

    def solve_step_null_input(self):

        for t in self.t_eval:

            # 1. Decide on your motor torque for this specific frame
            self.motor_force = 0

            # 2. Advance the physics by exactly one frame (dt)
            self.state = self.rk4_step()
            self.solution.append(np.append(self.state, 0))
            self.current_time += self.dt
            
        return np.array(self.solution), 0

    def solve_step_stablize_position(self, kp = 4, ki = 0.001, kd = 400):
        '''Just stabalize the cart at the desired location'''
        controller = pid.pid_controller(self.target, self.state[0], kp, ki, kd)
        cost = 0

        for t in self.t_eval:

            # 1. Decide on your motor torque for this specific frame
            controller.location = self.state[0]
            if t >= 2:
                self.motor_force = min((controller.update(), self.max_motor_force), key=abs)
            else:
                self.motor_force = 0
            # 2. Advance the physics by exactly one frame (dt)
            self.state = self.rk4_step()
            self.solution.append(np.append(self.state, self.motor_force))
            self.current_time += self.dt
            cost += (self.target - self.state[0])**2
            
        return np.array(self.solution), cost

    def solve_step_inverted_rod(self, kp = 50, ki = 0, kd = 1600, mode = 'analog', theta = 0):

        if mode == 'analog':
            angular_controller = pid.pid_controller(np.pi, self.state[1], kp, ki, kd)
            position_controller = pid.pid_controller(self.target, self.state[0], 13, 0, 1800, display = False)

        elif mode == 'RL':
            NN = nn.NeuralNetwork((4, 64, 64, 2), [nn.ReLU, nn.ReLU, [nn.linear, nn.sigmoid]], 'nn_library')
            NN.theta_recover(theta)
            
        cost = 0
        offset = 0
        discount = 0.2
        stable_counter = 0 
        state_history = []

        for t in self.t_eval:
            if mode == 'analog':
                
                angular_controller.location = self.state[1]
                position_controller.location = self.state[0]
                position_input = position_controller.update()
                angular_input = angular_controller.update()

                # Stage 0: If in excessive motion, stablalize
                if abs(self.state[3]) > 10 or abs(self.state[2]) > 10 or stable_counter == -1:
                    state_string = 'excessive motion'
                    self.motor_force = -(self.state[0] - self.target + self.state[2]) * self.max_motor_force * 0.1
                    stable_counter = -1
                    if abs(self.state[3]) < np.pi/2 and abs(self.state[2]) < 2:
                        stable_counter = 0

                # Stage 1: Initialize swing
                elif abs(self.state[1]) < 0.01 and abs(self.state[3]) < 0.01:
                    state_string = 'initialize swing'
                    self.motor_force = 10
                    stable_counter = 0
                
                # Stage 2: increase amplitude
                elif self.state[1] <= np.pi/2 or self.state[1] >= 3 * np.pi/2:
                    state_string = 'increase amplitude'
                    if self.state[3] > 0:
                        self.motor_force = -self.max_motor_force * math.cos(self.state[1])
                    else:
                        self.motor_force = self.max_motor_force * math.cos(self.state[1])
                    self.motor_force *= discount
                    stable_counter = 0
                
                # Stage 3: Kick to inverted position
                elif self.state[1] >= np.pi/2 and self.state[1] <= 3 * np.pi/2 and abs(self.state[1] - np.pi + 0.2 * math.atan(offset)) >= np.pi/5:
                    state_string = 'kick to inverted position'
                    angular_controller.kp = 5
                    angular_controller.kd = 200
                    angular_controller.target = np.pi
                    self.motor_force = angular_input
                    stable_counter = 0

                # Stage 4: Maintain 
                else:
                    state_string = 'maintain'
                    angular_controller.kp = kp
                    angular_controller.kd = kd
                    angular_controller.target = np.pi

                    stable_counter += 1 
                    if stable_counter >= 0.5 * self.fps:
                        offset = -0.1 * position_input
                        angular_controller.target = np.pi + 0.02 * math.atan(offset)

                    self.motor_force = angular_input
                
            elif mode == 'RL':
                scale_factors = np.array([10.0, 2 * np.pi, 50.0, 50.0])

                self.motor_force = (NN.feedforward(self.state / scale_factors)[-1][0][1] - 0.5) * 200

            # reject if the motor is asked to do more than it could
            if abs(self.motor_force) >= self.max_motor_force:
                print('Overload at ', t, 's')
                if self.motor_force > 0:
                    self.motor_force = self.max_motor_force
                else:
                    self.motor_force = -self.max_motor_force

            # 2. Advance the physics by exactly one frame (dt)
            self.state = self.rk4_step()
            self.solution.append(np.append(self.state, self.motor_force))
    
            if mode == 'analog':
                state_history.append(state_string)
            elif mode == "RL":
                state_history.append("RL")


            self.current_time += self.dt
            cost += (self.target - self.state[0])**2
            
        return np.array(self.solution), cost, state_history

    def animate(self, mode = 'RL', speed = 1, theta = 0):
        solution, cost, state_history = self.solve_step_inverted_rod(mode = mode, theta = theta)
        print('cost =', cost)
        # Extract position arrays for the animation
        x_cart_history = solution[:, 0]
        th1_history = solution[:, 1]
        force_history = solution[:, 4]

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-3, 7)

        # Setup empty artists
        cart_marker, = ax.plot([], [], 'ks', markersize=10) # Black square for cart
        rod1, = ax.plot([], [], 'b-', lw=2)
        rail = ax.plot([-30,30], [0, 0], 'k-', lw=2)

        # --- NEW: Setup the quiver object for the force arrow ---
        # scale_units='xy' and scale=1 means the arrow length matches plot coordinates.
        # We will manually scale the force magnitude below to keep it visually manageable.
        force_arrow = ax.quiver([0], [0], [0], [0], color='green', pivot='tail', 
                                angles='xy', scale_units='xy', scale=1, width=0.01, zorder=5)
        force_scale = 0.1 # Adjust this multiplier to change how long the arrow draws

        # 1. Setup the empty text object for the time
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12)
        location_text = ax.text(0.05, 0.8, '', transform=ax.transAxes, fontsize=12)
        angle_1_text = ax.text(0.05, 0.7, '', transform=ax.transAxes, fontsize=12)
        force_text = ax.text(0.05, 0.6, '', transform=ax.transAxes, fontsize=12)
        state_text = ax.text(0.05, 0.5, '', transform=ax.transAxes, fontsize=12)

        def init():
            cart_marker.set_data([], [])
            rod1.set_data([], [])
            
            # Reset the arrow to zero length at the origin
            force_arrow.set_offsets([[0, 0]])
            force_arrow.set_UVC(0, 0)
            
            # 2. Clear the text in the initialization
            time_text.set_text('')
            location_text.set_text('')
            angle_1_text.set_text('')
            force_text.set_text('')
            state_text.set_text('')
            
            # 3. Return the text artist alongside the others, including force_arrow
            return cart_marker, rod1, force_arrow, time_text, location_text, angle_1_text, force_text, state_text

        def update(frame):
            x_c = x_cart_history[frame]
            th1 = th1_history[frame]
            f = force_history[frame]
            state = state_history[frame]
            
            x1 = x_c + 1.0 * np.sin(th1)
            y1 = -1.0 * np.cos(th1)
                        
            cart_marker.set_data([x_c], [0])
            rod1.set_data([x_c, x1], [0, y1])
            
            # --- NEW: Update the force arrow ---
            # set_offsets sets the x,y starting coordinate of the arrow
            force_arrow.set_offsets([[x_c, 0]])
            # set_UVC sets the dx, dy vector components of the arrow
            force_arrow.set_UVC(f * force_scale, 0)
            
            # 4. Update the text string using the t_eval array
            current_time = self.t_eval[frame]
            time_text.set_text(f'Time: {current_time:.2f} s')
            location_text.set_text(f'Location: {x_c:.4f} m')
            angle_1_text.set_text(f'Angle 1: {th1:.4f} rad')
            force_text.set_text(f'Force: {f:.3f} N')
            state_text.set_text(f'State: {state}')
            
            # 5. Return the updated text artist
            return cart_marker, rod1, force_arrow, time_text, location_text, angle_1_text, force_text, state_text


        ani = animation.FuncAnimation(
            fig, update, frames=len(self.t_eval), 
            init_func=init, blit=True, interval=1000/self.fps * speed
        )
        plt.show()

if __name__ == "__main__":
    # SP = SinglePendulum(params=(9.8, 1, 1, 1), y0 = [0, 0, 0, 0],t_end=60)
    # SP.animate()

    # DP = DoublePendulum(params=(9.8, 1, 1, 1, 1, 1.2), t_end=60)
    # DP.animate()

    SP = SinglePendulum(params=(9.8, 1, 1, 1), y0 = [1, np.pi, 0, 0],t_end=60)
    SP.animate(mode = 'RL', theta = 2)
    SP = SinglePendulum(params=(9.8, 1, 1, 1), y0 = [-1, np.pi, 0, 0],t_end=60)
    SP.animate(mode = 'RL', theta = 2)

