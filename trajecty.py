import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math


class ReferenceTrajectoryGenerator:
    def __init__(self,
                 max_offset=2.0,  # Maximum lateral offset
                 period=400.0,  # Trajectory period length
                 smoothness=5.0,  # Smoothness factor
                 wheelbase=2.5,  # Vehicle wheelbase
                 max_accel=2.0,  # Maximum lateral acceleration
                 max_delta=0.5):  # Maximum steering angle (rad)

        self.max_lateral_offset = max_offset
        self.path_period = period
        self.smoothness_factor = smoothness
        self.vehicle_wheelbase = wheelbase
        self.max_lateral_accel = max_accel
        self.max_steering_angle = max_delta
        self.trajectory = []
        self.trajectory_generated = False

    def generateTrajectory(self, speed, duration, dt=0.01):
        # Clear old trajectory
        self.trajectory = []

        # Check parameter validity
        if speed <= 0 or duration <= 0 or dt <= 0:
            print("Invalid trajectory generation parameters")
            return False

        # Calculate safe speed and offset based on curvature and max lateral acceleration
        estimated_max_curvature = 4 * self.max_lateral_offset / (self.path_period ** 2)

        # Calculate safe maximum speed
        max_safe_speed = math.sqrt(self.max_lateral_accel / estimated_max_curvature)

        # If given speed exceeds safe speed, adjust speed
        actual_offset = self.max_lateral_offset
        actual_speed = speed

        if speed > max_safe_speed:
            # Reduce speed with 10% safety margin
            actual_speed = max_safe_speed * 0.9
            print(
                f"Warning: Speed too high for lateral acceleration constraint, adjusted to {actual_speed:.2f}m/s (original: {speed:.2f}m/s)")

        # Generate trajectory points using sine function for S-shape
        for t in np.arange(0, duration + dt, dt):
            x = actual_speed * t

            # Calculate normalized phase (0 to 2π)
            phase = 2 * math.pi * x / self.path_period

            # Generate S-shaped trajectory using sine
            y = actual_offset * math.sin(phase)

            # Calculate heading angle (psi) using analytical derivative
            dy_dx = actual_offset * 2 * math.pi / self.path_period * math.cos(phase)
            psi = math.atan2(dy_dx, 1.0)

            # Calculate curvature (kappa) using second derivative
            d2y_dx2 = -actual_offset * (2 * math.pi / self.path_period) ** 2 * math.sin(phase)
            kappa = d2y_dx2 / ((1 + dy_dx ** 2) ** (1.5))

            # Store trajectory point
            self.trajectory.append({
                'time': t,
                'x': x,
                'y': y,
                'psi': psi,
                'speed': actual_speed,
                'kappa': kappa
            })

        self.trajectory_generated = True
        print(
            f"Generated {len(self.trajectory)} trajectory points, total length: {self.trajectory[-1]['x']:.2f}m, duration: {self.trajectory[-1]['time']:.2f}s")
        return True

    def validateTrajectory(self):
        if not self.trajectory:
            return False

        valid = True
        violations = 0

        for i in range(1, len(self.trajectory)):
            point = self.trajectory[i]
            prev = self.trajectory[i - 1]

            # Check lateral acceleration constraint
            lateral_accel = point['speed'] ** 2 * abs(point['kappa'])
            if lateral_accel > self.max_lateral_accel:
                violations += 1
                valid = False

            # Check steering angle constraint
            steering = math.atan(self.vehicle_wheelbase * abs(point['kappa']))
            if steering > self.max_steering_angle:
                violations += 1
                valid = False

            # Check steering rate constraint
            dt = point['time'] - prev['time']
            if dt > 0:
                dkappa = abs(point['kappa'] - prev['kappa'])
                steering_rate = self.vehicle_wheelbase * dkappa * point['speed']
                if steering_rate > 0.1:  # Assume max steering rate is 0.1
                    violations += 1
                    valid = False

        if not valid:
            print(f"Warning: {violations} constraint violations detected.")
        else:
            print("Trajectory validation passed: All dynamic constraints satisfied!")

        return valid

    def saveTrajectoryToFile(self, filename):
        if not self.trajectory_generated or not self.trajectory:
            print("Error: Trajectory not generated or empty")
            return False

        with open(filename, 'w') as file:
            # Write header
            file.write("time,x,y,psi,speed,curvature\n")

            # Write data
            for point in self.trajectory:
                file.write(
                    f"{point['time']},{point['x']},{point['y']},{point['psi']},{point['speed']},{point['kappa']}\n")

        print(f"Trajectory saved to {filename}")
        return True

    def plotTrajectory(self, save_filename='D:/code/THU/ConsoleApplication1/trajectory_visualization.png'):
        if not self.trajectory_generated or not self.trajectory:
            print("Error: Trajectory not generated or empty")
            return

        # Extract data
        times = [point['time'] for point in self.trajectory]
        xs = [point['x'] for point in self.trajectory]
        ys = [point['y'] for point in self.trajectory]
        psis = [point['psi'] for point in self.trajectory]
        speeds = [point['speed'] for point in self.trajectory]
        kappas = [point['kappa'] for point in self.trajectory]

        # Calculate lateral acceleration
        lat_accels = [speeds[i] ** 2 * abs(kappas[i]) for i in range(len(speeds))]

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 2, figure=fig)

        # Trajectory plot (x-y plane)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(xs, ys, 'b-', linewidth=2)
        ax1.set_title('S-shaped Reference Trajectory')
        ax1.set_xlabel('Longitudinal Position (m)')
        ax1.set_ylabel('Lateral Position (m)')
        ax1.grid(True)

        # Lateral position vs time
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(times, ys, 'g-')
        ax2.set_title('Lateral Position vs Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Lateral Position (m)')
        ax2.grid(True)

        # Heading angle vs time
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(times, psis, 'r-')
        ax3.set_title('Heading Angle vs Time')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Heading Angle (rad)')
        ax3.grid(True)

        # Curvature vs time
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(times, kappas, 'm-')
        ax4.set_title('Curvature vs Time')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Curvature (1/m)')
        ax4.grid(True)

        # Lateral acceleration vs time
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(times, lat_accels, 'c-')
        ax5.axhline(y=self.max_lateral_accel, color='r', linestyle='--',
                    label=f'Max Limit: {self.max_lateral_accel} m/s²')
        ax5.set_title('Lateral Acceleration vs Time')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Lateral Acceleration (m/s²)')
        ax5.legend()
        ax5.grid(True)

        plt.tight_layout()
        plt.savefig(save_filename)
        print(f"Trajectory visualization saved to {save_filename}")

        return fig


# Generate and visualize trajectory
if __name__ == "__main__":
    # Create trajectory generator instance with appropriate parameters
    generator = ReferenceTrajectoryGenerator(
        max_offset=0.5,  # Maximum lateral offset
        period=200.0,  # Trajectory period - reduced to show multiple periods
        smoothness=5.0,  # Smoothness factor
        wheelbase=2.5,  # Wheelbase
        max_accel=2.0,  # Maximum lateral acceleration
        max_delta=0.5  # Maximum steering angle
    )

    # Generate trajectory
    sim_time = 20.0  # Simulation time (seconds)
    base_speed = 5.0  # Base speed (m/s) - increased to see more periods
    dt_traj = 0.01  # Trajectory time step

    generator.generateTrajectory(base_speed, sim_time, dt_traj)
    generator.validateTrajectory()

    # Save trajectory to file
    generator.saveTrajectoryToFile("generated_trajectory.csv")

    # Plot trajectory and save visualization
    generator.plotTrajectory()