import numpy as np
from simulator import Simulator
from pathlib import Path
import os
import pinocchio as pin
import matplotlib.pyplot as plt


# Load the robot model from scene XML
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
model = pin.buildModelFromMJCF(xml_path)
data = model.createData()

def plot_results(times: np.ndarray, positions:np.ndarray, velocities:np.ndarray, pos_err: np.ndarray, real: np.ndarray, estimated: np.ndarray, control: np.ndarray):
    """Plot and save simulation results."""

    # Joint positions plot
    plt.figure(figsize=(10, 6))
    for i in range(positions.shape[1]):
        plt.plot(times, positions[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Positions [rad]')
    plt.title('Joint Positions over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/damping_positions.png')
    plt.close()
    
    # Joint positions errors plot
    plt.figure(figsize=(10, 6))
    for i in range(pos_err.shape[1]):
        plt.plot(times, pos_err[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Position Errors [rad]')
    plt.title('Joint Position Errors over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/damping_position_errors.png')
    plt.close()

    # Joint velocities plot
    plt.figure(figsize=(10, 6))
    for i in range(velocities.shape[1]):
        plt.plot(times, velocities[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Velocities [rad/s]')
    plt.title('Joint Velocities over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/damping_velocities.png')
    plt.close()

    damping_parameters = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']
    link_parameters =  ['mass', 'mr_x', 'mr_y', 'mr_z', 'I_xx', 'I_xy', 'I_yy', 'I_xz', 'I_yz', 'I_zz']

    # Real link_parameters plot
    plt.figure(figsize=(10, 6))
    for i in range(real.shape[1]):
        plt.plot(times, real[:, i], label=link_parameters[i])
    plt.xlabel('Time [s]')
    plt.ylabel('Value of real parameter')
    plt.title('Value of real parameter over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/damping_real_link_parameters.png')
    plt.close()

    # Esimated link parameters plot
    plt.figure(figsize=(10, 6))
    for i in range(6, estimated.shape[1]):
        plt.plot(times, estimated[:, i], label=link_parameters[i-6])
    plt.xlabel('Time [s]')
    plt.ylabel('Value of estimated parameter')
    plt.title('Value of estimated parameter over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/damping_estimated_link_parameters.png')
    plt.close()

    # Esimated damping parameters plot
    plt.figure(figsize=(10, 6))
    for i in range(0, 6):
        plt.plot(times, estimated[:, i], label=damping_parameters[i])
    plt.xlabel('Time [s]')
    plt.ylabel('Value of estimated parameter')
    plt.title('Value of estimated parameter over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/damping_estimated_damping_parameters.png')
    plt.close()

    # Joint controls plot
    plt.figure(figsize=(10, 6))
    for i in range(control.shape[1]):
        plt.plot(times, control[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint control signals')
    plt.title('Joint control signals over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/damping_control_signals.png')
    plt.close()

def joint_controller(q: np.ndarray, dq: np.ndarray, p_hat, t: float) -> np.ndarray:
    """Joint space PD controller.
    
    Args:
        q: Current joint positions [rad]
        dq: Current joint velocities [rad/s]
        t: Current simulation time [s]
        
    Returns:
        tau: Joint torques command [Nm]
    """

    dt = 0.002

    q_des = np.array([-1.4, -1.5708, 1.5708, 0., 0., 0.]) # np.sin(2 * np.pi * t) in last link for trajectory
    
    dq_des = np.array([0., 0., 0., 0., 0., 0.]) # np.sin(2 * np.pi * t) in last link for trajectory
    
    ddq_des = np.array([0., 0., 0., 0., 0., 0.]) # np.sin(2 * np.pi * t) in last link for trajectory

    k = np.diag([20, 20, 40, 40, 30, 15])

    #errors
    q_err = q_des - q # position error
    print('q_err\n' , q_err)

    dq_err = dq_des - dq # velocities error

    lambd = 5
    s = dq_err + lambd * q_err # sliding surface

    dq_ref = dq_des + lambd * q_err # reference velocity
    ddq_ref = ddq_des + lambd * dq_err + k @ s # reference acceleration

    regressor = pin.computeJointTorqueRegressor(model, data, q, dq_ref, ddq_ref) # regression matrix of system dynamic 6x60

    regressor_damping = np.diag(-dq_ref) # damping regression matrix 6x6

    regressor_damping_with_6link_params = np.hstack([regressor_damping, regressor[:, 50:]]) # regressor matrix for unknown parameters
                                                                                            # including damping and 10 last parameters of last link 6x16

    regressor = np.hstack([regressor_damping, regressor]) # fully regression matrix with damping in begin and parameters of last link in the end 6x66
    
    gamma = 2500 # learning rate

    p_dot_hat = 1/gamma * regressor_damping_with_6link_params.T @ s # adaptive law

    p_hat = p_hat + dt * p_dot_hat # unknown parameters integration (prediction)

    state_vector = model.inertias[0].toDynamicParameters() # filling state vector with known parameters
    for i in range(1, 5):
        state_vector = np.hstack([state_vector, model.inertias[i].toDynamicParameters()])

    state_vector = np.hstack([p_hat[:6], state_vector, p_hat[6:]]) # column-vector state vector with predicted damping in begin
                                                                   # and last predicted parameters of last link in the end 66x1

    #print(state_vector)
    #print('p_hat\n', p_hat)
    #print('p_dot_hat', p_dot_hat)

    u = regressor @ state_vector # + k @ s # control law

    return u, p_hat, q_err

def main():

    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning real-time joint space control...")
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        record_video=True,
        video_path="logs/videos/adaptive_control_with_damping.mp4",
        width=1920,
        height=1080
    )
    sim.set_controller(joint_controller)

    sim.run(time_limit=5.0)

    plot_results(sim.times, sim.positions, sim.velocities, sim.pos_err, sim.real_state, sim.estimated_state, sim.controls) # graphs

if __name__ == "__main__":
    main() 