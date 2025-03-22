#include "vehicle_control_base.h"
#include <qpOASES.hpp>
#include <iomanip>
#include <algorithm>

// Include the trajectory generator and MPC controller cpp
#include "mpc_controller.cpp"
/**
 * 主函数 - 车辆控制仿真
 * 使用增强型轨迹生成器和改进的MPC控制器
 */
int main() {
    // 初始化车辆
    Vehicle vehicle;

    // 初始化增强型轨迹生成器
    ReferenceTrajectoryGenerator trajectory_generator(
        0.0,    // 最大横向偏移
        200.0,  // 轨迹周期长度
        5.0,    // 平滑度因子
        2.5,    // 轴距
        2.0,    // 最大横向加速度
        0.5,    // 最大转向角
        0.1     // 最大转向角速率
    );

    // 生成离线轨迹
    double sim_time = 20.0;  // 模拟时间（秒）
    double base_speed = 5.0; // 基准速度（m/s）
    double dt_traj = 0.01;   // 轨迹生成时间步长

    // 生成轨迹并保存
    trajectory_generator.generateTrajectory(base_speed, sim_time, dt_traj);
    trajectory_generator.saveTrajectoryToFile("generated_trajectory.csv");

    // 初始化MPC控制器 - 减少预测步数以提高性能
    ModelPredictiveController mpc(
        10,     // 预测步数 - 从30减少到15
        0.05    // 预测步长
    );

    // 调整MPC控制器权重以提高跟踪性能
    mpc.setWeights(
        2000.0,  // 位置误差权重 - 增加
        5000.0,  // 航向角误差权重 - 增加
        50.0,    // 横向速度误差权重
        30.0,    // 横摆角速度误差权重
        0.1,     // 转向角输入权重
        0.2      // 转向角变化率权重
    );

    // 初始化PID控制器（纵向速度控制）
    PID speed_pid(20, 0.02, 0.02, -2.0, 2.0);


    // 仿真参数
    double dt = 0.01;         // 时间步长
    int steps = static_cast<int>(sim_time / dt);

    // 参考速度
    double target_speed = base_speed;  // m/s

    // 统计指标
    double total_lateral_error = 0.0;
    double max_lateral_error = 0.0;
    double total_heading_error = 0.0;
    double max_heading_error = 0.0;
    double total_speed_error = 0.0;
    double max_speed_error = 0.0;

    // 初始化输出文件
    std::ofstream data_file("vehicle_control_data.csv");
    data_file << "time,x,y,vx,vy,psi,"
        << "y_ref,psi_ref,speed_ref,"
        << "lateral_error,heading_error,speed_error,"
        << "mpc_delta,pid_delta,"
        << "acceleration" << std::endl;

    // 上一步的转向角输入
    double prev_delta_f = 0.0;

    // 主循环前，初始化车辆位置到轨迹起点
    if (!trajectory_generator.trajectory.empty()) {
        const auto& start_point = trajectory_generator.getPointAtIndex(0);
        vehicle.x = start_point.x;
        vehicle.y = start_point.y;
        vehicle.psi = start_point.psi;
        vehicle.vx = base_speed * 0.0; // 初始速度设为目标速度的一半
    }

    std::cout << "开始仿真..." << std::endl;

    // 主循环
    size_t trajectory_index = 0; // 当前轨迹索引

    for (int i = 0; i < steps; ++i) {
        double t = i * dt;  // 当前仿真时间

        // 确保索引不超出轨迹长度
        if (trajectory_index >= trajectory_generator.trajectory.size()) {
            trajectory_index = trajectory_generator.trajectory.size() - 1;
        }

        // 直接使用索引获取参考点
        const auto& ref_point = trajectory_generator.getPointAtIndex(trajectory_index);
        double y_ref = ref_point.y;
        double psi_ref = ref_point.psi;
        double speed_ref = ref_point.speed;

        // 计算误差指标
        double lateral_error = vehicle.y - y_ref;
        double heading_error = vehicle.psi - psi_ref;
        while (heading_error > M_PI) heading_error -= 2 * M_PI;
        while (heading_error < -M_PI) heading_error += 2 * M_PI;

        double speed_error = vehicle.vx - target_speed;

        // 更新误差统计
        total_lateral_error += std::abs(lateral_error);
        max_lateral_error = std::max(max_lateral_error, std::abs(lateral_error));
        total_heading_error += std::abs(heading_error);
        max_heading_error = std::max(max_heading_error, std::abs(heading_error));
        total_speed_error += std::abs(speed_error);
        max_speed_error = std::max(max_speed_error, std::abs(speed_error));

        // MPC控制器计算转向角 - 传入当前索引
        double mpc_delta_f = mpc.compute(vehicle, trajectory_generator, prev_delta_f, trajectory_index);



        // 计算控制输入
        double max_speed_reduction = 2.0;  // 最大速度降低 2 m/s
        double error_threshold = 0.1;      // 横向误差阈值

        // 基于横向误差动态调整目标速度
        double speed_reduction = std::min(max_speed_reduction, std::abs(lateral_error) / error_threshold);
        double adjusted_target_speed = target_speed - speed_reduction;

        // 纵向控制：使用PID计算加速度
        double a = speed_pid.compute(adjusted_target_speed, vehicle.vx, dt);

        // 更新车辆状态
        vehicle.updateLongitudinal(a, dt);
        vehicle.updateLateral(mpc_delta_f, dt);  // 使用MPC的转向角

        // 更新上一步转向角
        prev_delta_f = mpc_delta_f;

        // 写入数据到CSV文件
        data_file << t << ","
            << vehicle.x << ","
            << vehicle.y << ","
            << vehicle.vx << ","
            << vehicle.vy << ","
            << vehicle.psi << ","
            << y_ref << ","
            << psi_ref << ","
            << target_speed << ","
            << lateral_error << ","
            << heading_error << ","
            << speed_error << ","
            << mpc_delta_f << ","
            << a << std::endl;

        // 每秒输出一次状态
        if (i % 100 == 0) {
            std::cout << "Time: " << std::fixed << std::setprecision(1) << t
                << "s, y: " << std::setprecision(2) << vehicle.y
                << "m, y_ref: " << y_ref
                << "m, err: " << lateral_error
                << "m, δ(MPC): " << mpc_delta_f
                << "rad, v: " << vehicle.vx << "m/s" << std::endl;
        }

        // 更新轨迹索引，根据实际速度和轨迹点间距确定增量
        // 这里假设等间距轨迹点，且以dt_traj为时间间隔生成
        double distance_traveled = vehicle.vx * dt;
        double distance_per_point = base_speed * dt_traj; // 轨迹点之间的距离
        int index_increment = static_cast<int>(std::round(distance_traveled / distance_per_point));

        // 确保至少前进一个点
        trajectory_index += std::max(1, index_increment);
    }

    // 计算并输出统计指标
    std::cout << "\n仿真性能统计:" << std::endl;
    std::cout << "平均横向误差: " << total_lateral_error / steps << "m" << std::endl;
    std::cout << "最大横向误差: " << max_lateral_error << "m" << std::endl;
    std::cout << "平均航向角误差: " << total_heading_error / steps << "rad" << std::endl;
    std::cout << "最大航向角误差: " << max_heading_error << "rad" << std::endl;
    std::cout << "平均速度误差: " << total_speed_error / steps << "m/s" << std::endl;
    std::cout << "最大速度误差: " << max_speed_error << "m/s" << std::endl;

    data_file.close();

    std::cout << "\n仿真完成！数据已保存到 vehicle_control_data.csv" << std::endl;
    std::cout << "轨迹数据已保存到 generated_trajectory.csv" << std::endl;

    return 0;
}