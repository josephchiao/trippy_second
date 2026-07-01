
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import physics
import controller

def DP_animate(solution, cost, state_history, params, t_eval, fps = 60, speed = 1):

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
        
        x1 = x_c + params[4] * np.sin(th1)
        y1 = -params[4] * np.cos(th1)
        
        x2 = x1 + params[5] * np.sin(th2)
        y2 = y1 - params[5] * np.cos(th2)
        
        cart_marker.set_data([x_c], [0])
        rod1.set_data([x_c, x1], [0, y1])
        rod2.set_data([x1, x2], [y1, y2])
        
        force_arrow.set_offsets([[x_c, 0]])
        force_arrow.set_UVC(f * force_scale, 0)
        
        current_time = t_eval[frame]
        time_text.set_text(f'Time: {current_time:.2f} s')
        location_text.set_text(f'Location: {x_c:.2f} m')
        angle_1_text.set_text(f'Angle 1: {th1:.2f} rad')
        angle_2_text.set_text(f'Angle 2: {th2:.2f} rad')
        force_text.set_text(f'Force: {f:.2f} N')
        state_text.set_text(f'State: {state}')
        
        return cart_marker, rod1, rod2, force_arrow, time_text, location_text, angle_1_text, angle_2_text, force_text, state_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(t_eval), 
        init_func=init, 
        blit=False, # <-- CRITICAL: Change blit to False to allow axis updates
        interval=1000/fps*speed
    )
    plt.show()

def SP_animate(solution, cost, state_history, params, t_eval, fps = 60, speed = 1):
    print('cost =', cost)
    # Extract position arrays for the animation
    x_cart_history = solution[:, 0]
    th1_history = solution[:, 1]
    force_history = solution[:, 4]

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-10, 15)
    ax.set_ylim(-3, 8)

    # Setup empty artists
    cart_marker, = ax.plot([], [], 'ks', markersize=10) # Black square for cart
    rod1, = ax.plot([], [], 'b-', lw=2)
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
        
        x1 = x_c + params[3] * np.sin(th1)
        y1 = -params[3] * np.cos(th1)
        
        cart_marker.set_data([x_c], [0])
        rod1.set_data([x_c, x1], [0, y1])
        
        force_arrow.set_offsets([[x_c, 0]])
        force_arrow.set_UVC(f * force_scale, 0)
        
        current_time = t_eval[frame]
        time_text.set_text(f'Time: {current_time:.2f} s')
        location_text.set_text(f'Location: {x_c:.2f} m')
        angle_1_text.set_text(f'Angle 1: {th1:.2f} rad')
        force_text.set_text(f'Force: {f:.2f} N')
        state_text.set_text(f'State: {state}')
        
        return cart_marker, rod1, force_arrow, time_text, location_text, angle_1_text, force_text, state_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(t_eval), 
        init_func=init, 
        blit=False, # <-- CRITICAL: Change blit to False to allow axis updates
        interval=1000/fps*speed
    )
    plt.show()



runtime = 60
refresh_rate = 60

def run(control_type = "None", animation = True, speed = 1):
    
    t = 0
    double_pendulum = physics.DoublePendulum(params = (9.81, 1, 1, 1, 1, 1.2), y0 = [0, 0, 0, 0, 0, 0], refresh_rate = refresh_rate)
    motor_controller = controller.DP_Controller(double_pendulum, target = 0, max_motor_force = 100)
    solution = []
    state_history = []
    total_cost = 0

    while t < runtime:
        t += 1/refresh_rate
        
        if control_type == "None":
            motor_force, state_string, cost = 0, 'None', 0
        elif control_type == "position_hold":
            motor_force, state_string, cost = motor_controller.solve_step_stablize_position()
        elif control_type == "inverted_rod_1":
            motor_force, state_string, cost = motor_controller.solve_step_inverted_rod_1()
        elif control_type == "inverted_rod_2":
            motor_force, state_string, cost = motor_controller.solve_step_inverted_rod_2()
        else:
            raise ValueError("Invalid control type. Choose from 'None', 'position_hold', 'inverted_rod_1', or 'inverted_rod_2'.")
        
        double_pendulum.motor_force = motor_force
        double_pendulum.rk4_step()
        state_history.append(state_string)
        solution.append(np.append(double_pendulum.state, double_pendulum.motor_force))
        total_cost += cost

    solution = np.array(solution)
    if animation:
        t_array = np.linspace(0, runtime, int(runtime * refresh_rate))
        DP_animate(solution, total_cost, state_history, double_pendulum.params, t_array, fps = refresh_rate, speed = speed)

def custom_run(control_type = [], time_table = [], animation = True, speed = 1):
    
    t = 0
    double_pendulum = physics.DoublePendulum(params = (9.81, 1, 1, 1, 1, 1.2), y0 = [3, 0, 0, 0, 0, 0], refresh_rate = refresh_rate)
    motor_controller = controller.DP_Controller(double_pendulum, target = 0, max_motor_force = 100)
    solution = []
    state_history = []
    total_cost = 0
    time_table.sort()

    if len(control_type) != len(time_table):
        raise ValueError("Invalid table")

    for stage in range(len(time_table)):
        
        while t < time_table[stage]:
            t += 1/refresh_rate
            
            if control_type[stage] == "None":
                motor_force, state_string, cost = 0, 'None', 0
            elif control_type[stage] == "position_hold":
                motor_force, state_string, cost = motor_controller.solve_step_stablize_position()
            elif control_type[stage] == "inverted_rod_1":
                motor_force, state_string, cost = motor_controller.solve_step_inverted_rod_1()
            elif control_type[stage] == "inverted_rod_2":
                motor_force, state_string, cost = motor_controller.solve_step_inverted_rod_2()
            else:
                raise ValueError("Invalid control type. Choose from 'None', 'position_hold', 'inverted_rod_1', or 'inverted_rod_2'.")
            
            double_pendulum.motor_force = motor_force
            double_pendulum.rk4_step()
            state_history.append(state_string)
            solution.append(np.append(double_pendulum.state, double_pendulum.motor_force))
            total_cost += cost


    solution = np.array(solution)
    if animation:
        t_array = np.linspace(0, runtime, int(time_table[-1] * refresh_rate))
        DP_animate(solution, total_cost, state_history, double_pendulum.params, t_array, fps = refresh_rate, speed = speed)

def SP_run(control_type = "None", animation = True, speed = 1):
    
    t = 0
    single_pendulum = physics.SinglePendulum(params = (9.81, 1, 1, 1), y0 = [0, np.pi, 0, 0], refresh_rate = refresh_rate)
    motor_controller = controller.SP_Controller(single_pendulum, target = 0, max_motor_force = 100)
    solution = []
    state_history = []
    total_cost = 0

    while t < runtime:
        t += 1/refresh_rate
        
        if control_type == "None":
            motor_force, state_string, cost = 0, 'None', 0
        elif control_type == "position_hold":
            motor_force, state_string, cost = motor_controller.solve_step_stablize_position()
        elif control_type == "inverted_rod":
            motor_force, state_string, cost = motor_controller.solve_step_inverted_rod()
        elif control_type == "ML":
            motor_force, state_string, cost = motor_controller.solve_step_inverted_rod(mode = 'ML')
        else:
            raise ValueError("Invalid control type. Choose from 'None', 'position_hold', 'inverted_rod_1', or 'inverted_rod_2'.")
        
        single_pendulum.motor_force = motor_force
        single_pendulum.rk4_step()
        state_history.append(state_string)
        solution.append(np.append(single_pendulum.state, single_pendulum.motor_force))
        total_cost += cost

    solution = np.array(solution)
    if animation:
        t_array = np.linspace(0, runtime, int(runtime * refresh_rate))
        SP_animate(solution, total_cost, state_history, single_pendulum.params, t_array, fps = refresh_rate, speed = speed)

if __name__ == "__main__":
    # run(control_type = "inverted_rod_2", animation = True, speed = 1)
    # custom_run(['position_hold', 'inverted_rod_1', 'inverted_rod_2'], [15,45,75])
    SP_run(control_type = "ML", animation = True, speed = 1)