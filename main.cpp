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
    vehicle.vx = 0.0; // 初始速度设为0

    const double DT_BASE = 0.01;           // 基础时间步长(秒)
    const double DT_CONTROL = DT_BASE;     // 控制更新步长
    const double DT_TRAJ = DT_BASE;        // 轨迹生成步长
    const double DT_MPC = DT_BASE;         // MPC预测步长也设为0.01秒
    const int MPC_HORIZON = 5;

    // 初始化PID控制器（纵向速度控制）
    PID speed_pid(20.0, 0.05, 0.02, -2.0, 2.0);

    // 初始化增强型轨迹生成器 - 传入PID和车辆对象
    ReferenceTrajectoryGenerator trajectory_generator(
        speed_pid,      // 速度PID控制器
        vehicle,        // 车辆模板
        1.0,            // 最大横向偏移
        200.0,          // 轨迹周期长度
        5.0,            // 平滑度因子
        2.5,            // 轴距
        2.0,            // 最大横向加速度
        0.5,            // 最大转向角
        0.1             // 最大转向角速率
    );

    // 生成离线轨迹 - 现在会考虑加速过程
    double sim_time = 20.0;  // 模拟时间（秒）
    double base_speed = 5.0; // 目标速度（m/s）
    double dt_traj = DT_BASE;   // 轨迹生成时间步长
    

    // 生成轨迹并保存
    trajectory_generator.generateTrajectory(base_speed, sim_time + 20.0, dt_traj);
    trajectory_generator.saveTrajectoryToFile("generated_trajectory.csv");

    // 初始化MPC控制器
    ModelPredictiveController mpc(
        MPC_HORIZON,     // 预测步数
        DT_MPC    // 预测步长
    );

    // 增强MPC控制器的权重
    mpc.setWeights(
        0.0,   // 位置误差权重
        0.0,   // 航向角误差权重
        0.0,     // 横向速度误差权重
        0.0,      // 横摆角速度误差权重
        50.0,       // 转向角输入权重
        50.0        // 转向角变化率权重
    );



    // 仿真参数
    double dt = DT_BASE;         // 时间步长
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

    // 重置车辆状态，准备实际仿真
    vehicle.x = 0.0;
    vehicle.y = 0.0;
    vehicle.psi = 0.0;
    vehicle.vx = 0.0;
    vehicle.vy = 0.0;
    vehicle.r = 0.0;

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

        double speed_error = vehicle.vx - speed_ref;

        // 更新误差统计
        total_lateral_error += std::abs(lateral_error);
        max_lateral_error = std::max(max_lateral_error, std::abs(lateral_error));
        total_heading_error += std::abs(heading_error);
        max_heading_error = std::max(max_heading_error, std::abs(heading_error));
        total_speed_error += std::abs(speed_error);
        max_speed_error = std::max(max_speed_error, std::abs(speed_error));

        // MPC控制器计算转向角 - 传入当前索引
        double mpc_delta_f = mpc.compute(vehicle, trajectory_generator, prev_delta_f, trajectory_index);


        // 纵向控制：使用PID计算加速度，参考速度现在使用轨迹点中的速度
        double a = speed_pid.compute(speed_ref, vehicle.vx, dt);

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
            << speed_ref << ","
            << lateral_error << ","
            << heading_error << ","
            << speed_error << ","
            << mpc_delta_f << ","
            << a << std::endl;

        if (i % 100 == 0) {
            std::cout << "Time: " << std::fixed << std::setprecision(1) << t
                << "s, y: " << std::setprecision(2) << vehicle.y
                << "m, y_ref: " << y_ref
                << "m, err: " << lateral_error
                << "m, δ(MPC): " << mpc_delta_f
                << "rad, v: " << vehicle.vx << "m/s" << std::endl;
        }

        // 根据x位置寻找下一个合适的轨迹点
        double min_dist = std::numeric_limits<double>::max();
        size_t closest_idx = trajectory_index;

        // 在未来一小段轨迹中查找最接近的点
        for (size_t j = trajectory_index; j < std::min(trajectory_index + 50, trajectory_generator.trajectory.size()); ++j) {
            double dist = std::abs(trajectory_generator.trajectory[j].x - vehicle.x);
            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = j;
            }
        }

        // 更新轨迹索引
        trajectory_index = closest_idx + 1; // 前进到下一个点
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