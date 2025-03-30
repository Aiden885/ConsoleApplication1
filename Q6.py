import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端以支持交互
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 定义车辆状态类
class VehicleState:
    def __init__(self, x=0.0, y=0.0, heading=0.0, velocity=0.0, current_lane=0):
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = velocity
        self.current_lane = current_lane

# 定义车道参数类
class LaneParams:
    def __init__(self, width=4.0):
        self.width = width

# 定义轨迹类
class Trajectory:
    def __init__(self):
        self.points = []  # 存储 (x, y) 点的列表

# 定义贝塞尔曲线计算类
class BezierCurve:
    @staticmethod
    def calculate_cubic_bezier(control_points, num_points):
        """计算三次贝塞尔曲线"""
        t = np.linspace(0, 1, num_points)
        points = []
        for ti in t:
            x = ((1-ti)**3 * control_points[0][0] +
                 3*(1-ti)**2 * ti * control_points[1][0] +
                 3*(1-ti) * ti**2 * control_points[2][0] +
                 ti**3 * control_points[3][0])
            y = ((1-ti)**3 * control_points[0][1] +
                 3*(1-ti)**2 * ti * control_points[1][1] +
                 3*(1-ti) * ti**2 * control_points[2][1] +
                 ti**3 * control_points[3][1])
            points.append((x, y))
        return points

# 定义轨迹生成类
class TrajGenerator:
    def __init__(self, lane_params=None):
        self.vehicle_state = VehicleState()
        self.lane_params = lane_params if lane_params else LaneParams()

    def set_vehicle_state(self, state):
        self.vehicle_state = state

    def generate_trajectories(self, planning_distance, num_trajectories, ctrl1_x_factor, ctrl1_y_factor, ctrl2_x_factor, ctrl2_y_factor):
        """生成轨迹（这里以车道内轨迹为例，可扩展为换道等）"""
        trajectories = []
        if num_trajectories % 2 == 0:
            num_trajectories += 1
        lane_center_y = self.lane_params.width / 2
        lateral_range = self.lane_params.width * 0.8
        lateral_step = lateral_range / num_trajectories

        for i in range(num_trajectories):
            trajectory = Trajectory()
            lateral_offset = -lateral_range / 2 + i * lateral_step
            end_y = lane_center_y + lateral_offset

            control_points = [
                (self.vehicle_state.x, self.vehicle_state.y),
                (self.vehicle_state.x + planning_distance * ctrl1_x_factor,
                 self.vehicle_state.y + (end_y - self.vehicle_state.y) * ctrl1_y_factor),
                (self.vehicle_state.x + planning_distance * ctrl2_x_factor,
                 end_y + (end_y - self.vehicle_state.y) * ctrl2_y_factor),
                (self.vehicle_state.x + planning_distance, end_y)
            ]

            num_points = 100
            trajectory.points = BezierCurve.calculate_cubic_bezier(control_points, num_points)
            trajectories.append(trajectory)

        return trajectories

# 初始化
vehicle_state = VehicleState(x=0.0, y=0.0, current_lane=0)
lane_params = LaneParams(width=4.0)
traj_gen = TrajGenerator(lane_params=lane_params)
traj_gen.set_vehicle_state(vehicle_state)

# 创建绘图窗口
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.25)  # 为滑块留出空间

# 初始参数
initial_planning_distance = 100.0
initial_num_trajectories = 5
initial_ctrl1_x = 0.1
initial_ctrl1_y = 0.4
initial_ctrl2_x = 0.9
initial_ctrl2_y = 0.3

# 绘制初始轨迹
trajectories = traj_gen.generate_trajectories(
    initial_planning_distance, initial_num_trajectories, initial_ctrl1_x, initial_ctrl1_y, initial_ctrl2_x, initial_ctrl2_y
)
lines = [ax.plot(*zip(*traj.points))[0] for traj in trajectories]

ax.set_title("Dynamic Trajectories")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.grid(True)
ax.axis('equal')

# 添加滑块
ax_dist = plt.axes([0.25, 0.05, 0.65, 0.03])
ax_ctrl1_x = plt.axes([0.25, 0.10, 0.65, 0.03])
ax_ctrl1_y = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_ctrl2_x = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_ctrl2_y = plt.axes([0.25, 0.25, 0.65, 0.03])

slider_dist = Slider(ax_dist, 'Distance', 50.0, 200.0, valinit=initial_planning_distance)
slider_ctrl1_x = Slider(ax_ctrl1_x, 'Ctrl1 X', 0.01, 0.5, valinit=initial_ctrl1_x)
slider_ctrl1_y = Slider(ax_ctrl1_y, 'Ctrl1 Y', 0.0, 1.0, valinit=initial_ctrl1_y)
slider_ctrl2_x = Slider(ax_ctrl2_x, 'Ctrl2 X', 0.5, 0.99, valinit=initial_ctrl2_x)
slider_ctrl2_y = Slider(ax_ctrl2_y, 'Ctrl2 Y', 0.0, 1.0, valinit=initial_ctrl2_y)

# 更新函数
def update(val):
    planning_distance = slider_dist.val
    ctrl1_x = slider_ctrl1_x.val
    ctrl1_y = slider_ctrl1_y.val
    ctrl2_x = slider_ctrl2_x.val
    ctrl2_y = slider_ctrl2_y.val

    trajectories = traj_gen.generate_trajectories(
        planning_distance, initial_num_trajectories, ctrl1_x, ctrl1_y, ctrl2_x, ctrl2_y
    )
    for i, line in enumerate(lines):
        line.set_data(*zip(*trajectories[i].points))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

# 绑定滑块事件
slider_dist.on_changed(update)
slider_ctrl1_x.on_changed(update)
slider_ctrl1_y.on_changed(update)
slider_ctrl2_x.on_changed(update)
slider_ctrl2_y.on_changed(update)

# 显示窗口
plt.show()