import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import heapq
from collections import defaultdict


class AutoParking:
    def __init__(self):
        # 初始化参数设置
        self.setup_parameters()
        # 创建可行驶空间
        self.create_drivable_space()
        # 创建画布和图形
        self.setup_figure()
        # 绘制环境元素
        self.draw_environment()
        # 计算泊车路径
        self.plan_path()
        # 初始化动画
        self.setup_animation()

    def setup_parameters(self):
        """设置基本参数"""
        # 边界设置
        self.x_min, self.x_max = -10, 15
        self.y_min, self.y_max = -4, 10

        # 停车位参数
        self.parking_spot = {'x': -5, 'y': 2, 'width': 3, 'length': 5}  # 车库：x=-5 到 x=0, y=2 到 y=5

        # 车辆参数
        self.vehicle = {'length': 4, 'width': 2, 'x': 5, 'y': -1, 'angle': np.pi / 2}  # 小车初始位置 (5, -2)，车头向上
        bias_angle = np.radians(-10)  # -10度的偏角（弧度制），负值表示向右偏
        self.vehicle['angle'] += bias_angle  # 加上右偏角度，确保车头偏右

        # 环境参数
        self.wall_x = 10  # 右侧墙位置
        self.left_wall_x = 0  # 左侧墙位置

        # 混合A*算法参数
        self.grid_size = 0.25  # 网格大小，减小以提高分辨率
        self.steering_angles = np.radians([-30, -15, 0, 15, 30])  # 离散转向角度
        self.wheel_base = 2.5  # 轴距
        self.min_turning_radius = self.wheel_base / np.tan(np.radians(30))  # 最小转弯半径
        self.step_size = 0.5  # 每步移动距离，减小以提高精度
        self.reverse_cost = 2.0  # 倒车成本
        self.forward_cost = 5.0  # 前进成本（前进代价高，优先倒车）
        self.steering_change_cost = 1.5  # 转向变化成本
        self.direction_change_cost = 5.0  # 方向变化成本
        self.safety_margin = 0.05  # 安全边距，减小以提高成功率

        # 路径点列表
        self.path_points = []

        # 轨迹记录
        self.trajectory_x = []
        self.trajectory_y = []

        # 碰撞检测参数
        self.collision_check_steps = 10  # 检查中间路径点的数量

    def create_drivable_space(self):
        """创建可驾驶空间"""
        # 1. 主区域: x∈[0,10], y∈[-4,10]
        main_area = {
            'x_min': 0,
            'x_max': 10,
            'y_min': -4,
            'y_max': 10
        }

        # 2. 车库区域: x∈[-5,0], y∈[2,5]
        garage_area = {
            'x_min': -5,
            'x_max': 0,
            'y_min': 2,
            'y_max': 5
        }

        # 存储可行区域
        self.drivable_areas = [main_area, garage_area]

    def setup_figure(self):
        """设置matplotlib图形"""
        # 强制使用 TkAgg 后端（解决某些显示问题）
        plt.switch_backend('TkAgg')

        # 设置支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 创建画布
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

        # 设置坐标轴范围
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        plt.title("基于混合A*算法的自动泊车演示：倒车入库")

        # 初始化车辆图形
        self.init_vehicle()

    def init_vehicle(self):
        """初始化车辆图形"""
        # 创建矩形表示车身
        self.rect = patches.Rectangle((0, 0), self.vehicle['length'], self.vehicle['width'],
                                      angle=0, edgecolor='blue', facecolor='none')
        self.ax.add_patch(self.rect)

        # 添加车辆前后轮指示器（可选）
        self.front_wheel = plt.Line2D([0, 0], [0, 0], color='red', lw=2)
        self.rear_wheel = plt.Line2D([0, 0], [0, 0], color='green', lw=2)
        self.ax.add_line(self.front_wheel)
        self.ax.add_line(self.rear_wheel)

        # 立即更新车辆初始位置显示
        self.update_vehicle_display()

    def draw_environment(self):
        """绘制环境元素（车库、墙壁等）"""
        # 绘制车库（三面有墙）
        # 车库底面（y=2）
        self.ax.plot([-5, 0], [2, 2], color='black', linewidth=2)
        # 车库顶面（y=5）
        self.ax.plot([-5, 0], [5, 5], color='black', linewidth=2)
        # 车库左侧墙（x=-5）
        self.ax.plot([-5, -5], [2, 5], color='black', linewidth=2)
        # 填充车库区域
        self.ax.add_patch(patches.Rectangle((self.parking_spot['x'], self.parking_spot['y']),
                                            self.parking_spot['length'], self.parking_spot['width'],
                                            edgecolor='none', facecolor='gray', alpha=0.3))

        # 绘制 x=0 处的墙（除了车库开口）
        # x=0, y=-4 到 y=2
        self.ax.plot([0, 0], [-4, 2], color='black', linewidth=2)
        # x=0, y=5 到 y=10
        self.ax.plot([0, 0], [5, 10], color='black', linewidth=2)

        # 绘制右侧墙
        self.ax.plot([self.wall_x, self.wall_x], [-4, 10], color='black', linewidth=2)

        # 可选：可视化可行驶空间
        self.visualize_drivable_space()

    def visualize_drivable_space(self):
        """可视化可行驶空间（用于调试）"""
        for i, area in enumerate(self.drivable_areas):
            color = 'green' if i == 0 else 'cyan'
            self.ax.add_patch(patches.Rectangle(
                (area['x_min'], area['y_min']),
                area['x_max'] - area['x_min'],
                area['y_max'] - area['y_min'],
                edgecolor=color, facecolor='none', linestyle='--', alpha=0.5
            ))

    def check_collision(self, x, y, angle, length, width):
        """检查车辆是否与墙壁发生碰撞"""
        # 计算车辆四个角的坐标
        half_length = length / 2
        half_width = width / 2

        # 计算旋转矩阵
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # 计算四个角的坐标（相对于车辆中心）
        corners_rel = [
            [-half_length, -half_width],  # 左后
            [half_length, -half_width],  # 左前
            [half_length, half_width],  # 右前
            [-half_length, half_width]  # 右后
        ]

        # 转换到世界坐标系
        corners = []
        for corner_rel in corners_rel:
            corner_x = x + corner_rel[0] * cos_angle - corner_rel[1] * sin_angle
            corner_y = y + corner_rel[0] * sin_angle + corner_rel[1] * cos_angle
            corners.append((corner_x, corner_y))

        # 添加车身中心点
        all_points = [(x, y)]  # 车辆中心点
        all_points.extend(corners)  # 四个角

        # 减少边界点数量，只在每条边中间添加一个点
        for i in range(4):
            j = (i + 1) % 4  # 下一个角的索引
            mid_x = (corners[i][0] + corners[j][0]) / 2  # 边的中点
            mid_y = (corners[i][1] + corners[j][1]) / 2
            all_points.append((mid_x, mid_y))

        # 安全边距 - 减小到很小的值
        safety_margin = 0.01  # 降低安全边距到极小

        # 获取两个可行区域
        main_area = self.drivable_areas[0]
        garage_area = self.drivable_areas[1]

        # 调试输出
        if x == self.vehicle['x'] and y == self.vehicle['y']:  # 如果是检查起点
            print("检查起点碰撞状态:")
            print(f"中心点: ({x:.2f}, {y:.2f})")
            for i, (px, py) in enumerate(corners):
                print(f"角点 {i + 1}: ({px:.2f}, {py:.2f})")

        # 检查每个点是否在可行区域内
        for i, (point_x, point_y) in enumerate(all_points):
            # 检查点是否在主区域内
            in_main_area = (
                    point_x >= main_area['x_min'] + safety_margin and
                    point_x <= main_area['x_max'] - safety_margin and
                    point_y >= main_area['y_min'] + safety_margin and
                    point_y <= main_area['y_max'] - safety_margin
            )

            # 检查点是否在车库区域内
            in_garage_area = (
                    point_x >= garage_area['x_min'] + safety_margin and
                    point_x <= garage_area['x_max'] - safety_margin and
                    point_y >= garage_area['y_min'] + safety_margin and
                    point_y <= garage_area['y_max'] - safety_margin
            )

            # 如果点既不在主区域内也不在车库区域内，则发生碰撞
            if not (in_main_area or in_garage_area):
                # 调试输出
                if x == self.vehicle['x'] and y == self.vehicle['y']:  # 如果是检查起点
                    point_type = "中心点" if i == 0 else f"角点 {i}" if i <= 4 else f"边点 {i - 4}"
                    print(f"碰撞点: {point_type} ({point_x:.2f}, {point_y:.2f})")
                    print(f"  在主区域内: {in_main_area}")
                    print(f"  在车库区域内: {in_garage_area}")
                return True

        # 所有点都在可行区域内，没有碰撞
        return False

    def check_path_collision(self, start_state, end_state, steer, direction):
        """检查两个状态之间的路径是否有碰撞"""
        x1, y1, angle1 = start_state
        x2, y2, angle2 = end_state

        # 在路径上检查多个中间点
        for i in range(1, self.collision_check_steps):
            t = i / self.collision_check_steps

            # 简单线性插值
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            angle = angle1 + t * (angle2 - angle1)

            if self.check_collision(x, y, angle, self.vehicle['length'], self.vehicle['width']):
                return True

        return False

    def plan_path(self):
        """使用混合A*算法规划泊车路径"""
        print("使用混合A*算法规划路径...")

        # 起点和终点状态
        start_state = (self.vehicle['x'], self.vehicle['y'], self.vehicle['angle'])

        # 目标状态：车头向右（0度）
        goal_state = (self.parking_spot['x'] + self.parking_spot['length'] / 2,
                      self.parking_spot['y'] + self.parking_spot['width'] / 2,
                      0.0)  # 车头朝右（与x轴正方向平行）

        # 打印起点和终点状态
        print(f"起点: x={start_state[0]:.2f}, y={start_state[1]:.2f}, θ={np.degrees(start_state[2]):.2f}度")
        print(f"终点: x={goal_state[0]:.2f}, y={goal_state[1]:.2f}, θ={np.degrees(goal_state[2]):.2f}度")

        # 检查起点是否在可行区域内
        if self.check_collision(start_state[0], start_state[1], start_state[2],
                                self.vehicle['length'], self.vehicle['width']):
            print("警告：起点被判定为碰撞状态！")

        # 检查终点是否在可行区域内
        if self.check_collision(goal_state[0], goal_state[1], goal_state[2],
                                self.vehicle['length'], self.vehicle['width']):
            print("警告：终点被判定为碰撞状态！")

        # 混合A*搜索
        self.path_points = self.hybrid_a_star(start_state, goal_state)

        if self.path_points:
            print(f"路径规划成功，共{len(self.path_points)}个路径点")
            # 顺序反转（从起点到终点）
            self.path_points.reverse()
        else:
            print("路径规划失败：无法找到从起点到终点的有效路径。")
            print("可能的原因：")
            print("1. 车库空间不足")
            print("2. 障碍物太多")
            print("3. 车辆转向半径限制")
            print("4. 算法参数需要调整")

    def hybrid_a_star(self, start, goal):
        """混合A*算法实现"""
        print("开始混合A*搜索...")

        # 优先队列
        open_list = []
        # 已访问节点（使用栅格坐标和离散角度作为键）
        closed_set = set()
        # 节点信息（保存父节点和路径代价）
        node_info = {}

        # 网格分辨率和角度分辨率
        xy_resolution = self.grid_size
        angle_resolution = np.radians(10)  # 角度分辨率（10度）

        # 打印可行区域信息
        print("可行区域定义:")
        for i, area in enumerate(self.drivable_areas):
            print(f"区域 {i + 1}: x∈[{area['x_min']}, {area['x_max']}], y∈[{area['y_min']}, {area['y_max']}]")

        # 初始化起点
        heapq.heappush(open_list, (0, start))  # (估计总成本, 状态)
        node_info[start] = {'parent': None, 'cost': 0, 'steer': 0, 'direction': 1}

        # 搜索迭代次数限制
        max_iterations = 1000000
        iterations = 0

        # 记录第一次迭代的扩展信息
        first_iteration_expansions = []

        while open_list and iterations < max_iterations:
            iterations += 1

            # 获取当前代价最小的节点
            _, current = heapq.heappop(open_list)
            current_x, current_y, current_angle = current

            # 打印搜索进度
            if iterations == 1 or iterations % 100 == 0:
                print(
                    f"搜索迭代: {iterations}, 当前位置: ({current_x:.2f}, {current_y:.2f}), 角度: {np.degrees(current_angle):.1f}度")

            # 将当前节点添加到已访问集合
            grid_x = round(current_x / xy_resolution) * xy_resolution
            grid_y = round(current_y / xy_resolution) * xy_resolution
            grid_angle = round(current_angle / angle_resolution) * angle_resolution
            grid_key = (grid_x, grid_y, grid_angle)

            if grid_key in closed_set:
                continue

            closed_set.add(grid_key)

            # 检查是否到达目标
            if self.is_goal(current, goal):
                print(f"找到目标! 迭代次数: {iterations}")
                # 重建路径
                return self.reconstruct_path(node_info, current)

            # 跟踪第一次迭代的扩展结果
            valid_expansions = 0
            total_expansions = 0

            # 对每个可能的转向角度和方向进行扩展
            for steer in self.steering_angles:
                for direction in [-1, 1]:  # -1表示倒车，1表示前进
                    total_expansions += 1

                    # 模拟车辆运动，获取下一个状态
                    next_x, next_y, next_angle = self.move_vehicle(
                        current_x, current_y, current_angle, steer, direction, self.step_size)
                    next_state = (next_x, next_y, next_angle)

                    # 记录第一次迭代的扩展详情
                    if iterations == 1:
                        collision = self.check_collision(next_x, next_y, next_angle,
                                                         self.vehicle['length'], self.vehicle['width'])
                        path_collision = self.check_path_collision(current, next_state, steer, direction)

                        first_iteration_expansions.append({
                            'steer': np.degrees(steer),
                            'direction': "倒车" if direction == -1 else "前进",
                            'next_state': (next_x, next_y, np.degrees(next_angle)),
                            'collision': collision,
                            'path_collision': path_collision
                        })

                    # 检查是否碰撞
                    if self.check_collision(next_x, next_y, next_angle, self.vehicle['length'], self.vehicle['width']):
                        continue

                    # 检查路径碰撞
                    if self.check_path_collision(current, next_state, steer, direction):
                        continue

                    valid_expansions += 1

                    # 计算各种成本
                    move_cost = self.reverse_cost if direction == -1 else self.forward_cost

                    if current in node_info:
                        prev_steer = node_info[current]['steer']
                        steer_change_cost = abs(steer - prev_steer) * self.steering_change_cost

                        prev_direction = node_info[current]['direction']
                        direction_change_cost = 0.0
                        if prev_direction != direction:
                            direction_change_cost = self.direction_change_cost
                    else:
                        steer_change_cost = 0.0
                        direction_change_cost = 0.0

                    next_cost = node_info[current][
                                    'cost'] + move_cost + steer_change_cost + direction_change_cost if current in node_info else move_cost

                    grid_next_key = (round(next_x / xy_resolution) * xy_resolution,
                                     round(next_y / xy_resolution) * xy_resolution,
                                     round(next_angle / angle_resolution) * angle_resolution)

                    if grid_next_key in closed_set:
                        continue

                    h_cost = self.calculate_heuristic(next_state, goal)
                    total_cost = next_cost + h_cost

                    if next_state not in node_info or next_cost < node_info[next_state]['cost']:
                        node_info[next_state] = {
                            'parent': current,
                            'cost': next_cost,
                            'steer': steer,
                            'direction': direction
                        }
                        heapq.heappush(open_list, (total_cost, next_state))

            # 在第一次迭代后打印详细信息
            if iterations == 1:
                print(f"第一次迭代扩展结果: 有效扩展 {valid_expansions}/{total_expansions}")
                print("详细扩展信息:")
                for i, exp in enumerate(first_iteration_expansions):
                    print(f"{i + 1}. 转向: {exp['steer']}度, 方向: {exp['direction']}, "
                          f"下一状态: {exp['next_state']}, "
                          f"碰撞: {'是' if exp['collision'] else '否'}, "
                          f"路径碰撞: {'是' if exp['path_collision'] else '否'}")

        print(f"搜索结束，未找到路径。迭代次数: {iterations}")
        return []

    def is_goal(self, current, goal):
        """判断是否到达目标状态的附近"""
        x_tolerance = 0.5
        y_tolerance = 0.5
        angle_tolerance = np.radians(15)

        x_diff = abs(current[0] - goal[0])
        y_diff = abs(current[1] - goal[1])

        # 计算角度差，考虑周期性
        angle_diff = abs((current[2] - goal[2] + np.pi) % (2 * np.pi) - np.pi)

        return x_diff < x_tolerance and y_diff < y_tolerance and angle_diff < angle_tolerance

    def calculate_heuristic(self, state, goal):
        """计算启发式代价（考虑非完整约束）"""
        # 欧氏距离
        dist = np.sqrt((state[0] - goal[0]) ** 2 + (state[1] - goal[1]) ** 2)

        # 角度差异
        angle_diff = abs((state[2] - goal[2] + np.pi) % (2 * np.pi) - np.pi)

        # 考虑转向约束的估计代价
        # 简化版本的Reeds-Shepp距离估计
        min_radius = self.wheel_base / np.tan(np.radians(30))
        min_turn_dist = min_radius * angle_diff

        return dist + min_turn_dist * 0.5

    def move_vehicle(self, x, y, angle, steer, direction, distance):
        """基于自行车模型模拟车辆运动"""
        # 自行车模型
        # x' = x + distance * cos(angle)
        # y' = y + distance * sin(angle)
        # angle' = angle + distance * tan(steer) / wheel_base

        # 调整方向（前进或后退）
        distance *= direction

        # 计算下一个状态
        next_x = x + distance * np.cos(angle)
        next_y = y + distance * np.sin(angle)
        next_angle = (angle + distance * np.tan(steer) / self.wheel_base) % (2 * np.pi)

        return next_x, next_y, next_angle

    def reconstruct_path(self, node_info, goal_state):
        """从目标状态回溯到起点，重建路径"""
        path = [goal_state]
        current = goal_state

        while current in node_info and node_info[current]['parent'] is not None:
            current = node_info[current]['parent']
            path.append(current)

        return path

    def update(self, frame):
        """动画更新函数"""
        # 检查路径点是否为空
        if not self.path_points:
            print("没有有效路径，动画结束。")
            return self.rect, self.front_wheel, self.rear_wheel

        # 保存上一帧位置用于检测碰撞
        old_x, old_y, old_angle = self.vehicle['x'], self.vehicle['y'], self.vehicle['angle']

        # 获取当前帧的位置和角度
        if frame < len(self.path_points):
            x, y, angle = self.path_points[frame]

            # 更新车辆状态
            self.vehicle['x'] = x
            self.vehicle['y'] = y
            self.vehicle['angle'] = angle

            # 打印位置更新（每10帧打印一次）
            if frame % 10 == 0:
                print(f"帧 {frame}: 位置 ({x:.2f}, {y:.2f}), 角度 {np.degrees(angle):.2f}度")

            # 检查是否会碰撞墙壁，如果会则保持原位置
            if self.check_collision(x, y, angle, self.vehicle['length'], self.vehicle['width']):
                print(f"警告：帧 {frame} 检测到碰撞，保持原位置")
                self.vehicle['x'], self.vehicle['y'], self.vehicle['angle'] = old_x, old_y, old_angle

            # 更新轨迹和车辆图形
            self.update_trajectory()
            self.update_vehicle_display()

        return self.rect, self.front_wheel, self.rear_wheel

    def update_trajectory(self):
        """更新并绘制轨迹"""
        # 保存轨迹点
        self.trajectory_x.append(self.vehicle['x'])
        self.trajectory_y.append(self.vehicle['y'])

        # 绘制轨迹
        if len(self.trajectory_x) > 1:
            self.ax.plot(self.trajectory_x[-2:], self.trajectory_y[-2:], 'g-', alpha=0.5)

    def update_vehicle_display(self):
        """更新车辆显示"""
        # 车身中心坐标
        x = self.vehicle['x']
        y = self.vehicle['y']
        angle = self.vehicle['angle']
        length = self.vehicle['length']
        width = self.vehicle['width']

        # 计算车辆四个角的坐标（考虑旋转）
        half_length = length / 2
        half_width = width / 2

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # 计算四个角的坐标（相对于车辆中心）
        bl_x = x - half_length * cos_angle - half_width * sin_angle  # 左下
        bl_y = y - half_length * sin_angle - half_width * cos_angle

        # 设置矩形位置为左下角坐标，同时保持长宽不变
        self.rect.set_xy((bl_x, bl_y))

        # 设置矩形角度
        self.rect.set_angle(np.degrees(angle))

    def setup_animation(self):
        """设置并启动动画"""
        if not self.path_points:
            print("警告：没有找到有效路径，无法创建动画。")
            total_frames = 1  # 设置最小帧数，避免错误
        else:
            total_frames = len(self.path_points)
            print(f"设置动画，共{total_frames}帧")

        # 创建动画
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=total_frames,
            interval=50,
            blit=True,
            init_func=lambda: (self.rect, self.front_wheel, self.rear_wheel)
        )

    def show(self):
        """显示动画"""
        plt.show()


# 主函数
def main():
    print("开始执行自动泊车程序")
    auto_parking = AutoParking()
    auto_parking.show()


if __name__ == "__main__":
    main()