import math
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import sympy as sm
import pid 

class DoublePendulum:
    """
    A class representing a Double Pendulum on a cart system.
    This class encapsulates the physics engine and numerical integration
    to simulate the dynamics of the system.

    angle = 0 at straight down.
    """

    def __init__(self, params = (9.81, 1, 1, 1, 1, 1), y0 = [0, 0, 0, 0, 0, 0], refresh_rate = 60):
        self.calc_M, self.calc_F = self.physics_engine()
        self.params = params  # gravity, mass of cart, mass rod1, mass rod2, length rod1, length rod2
        self.state = y0  # location of cart, angle rod1, angle rod2, velocity of cart, angular velocity rod1, angular velocity rod2
        self.motor_force = 0
        self.refresh_rate = refresh_rate          # Animation frames per second
        self.dt = 1.0 / self.refresh_rate

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
        
        self.state = new_state  # Update the state of the system
        return self.state


class SinglePendulum:
    """
    A class representing a single pendulum on a cart system.
    This class encapsulates the physics engine and numerical integration
    to simulate the dynamics of the system.

    angle = 0 at straight down.
    """

    def __init__(self, params = (9.81, 1, 1, 1), y0 = [0, 0, 0, 0], refresh_rate = 60):
        self.calc_M, self.calc_F = self.physics_engine()
        self.params = params  # gravity, mass of cart, mass rod1, mass rod2, length rod1, length rod2
        self.state = y0  # location of cart, angle rod1, angle rod2, velocity of cart, angular velocity rod1, angular velocity rod2
        self.motor_force = 0
        self.refresh_rate = refresh_rate          # Animation frames per second
        self.dt = 1.0 / self.refresh_rate

    def physics_engine(self):
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

        
        self.state = new_state  # Update the state of the system
        return self.state