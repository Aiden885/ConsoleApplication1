#ifndef VEHICLE_CONTROL_BASE_H
#define VEHICLE_CONTROL_BASE_H

#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cstdlib>

// 确保M_PI在所有编译环境下可用
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace Eigen;
using namespace std;

// 参考点结构体
struct ReferencePoint {
    double time;     // 时间
    double x;        // x位置
    double y;        // y位置
    double psi;      // 航向角
    double speed;    // 速度
};

/**
 * PID控制器类
 * 用于基本的反馈控制
 */
class PID {
private:
    // PID参数
    double Kp;         // 比例系数
    double Ki;         // 积分系数
    double Kd;         // 微分系数

    // 状态变量
    double integral;           // 积分项
    double previous_error;     // 上一次误差
    double previous_input;     // 上一次控制输入

    // 输出限制
    double output_min;         // 最小输出
    double output_max;         // 最大输出

    // 抗积分饱和参数
    double windup_guard;       // 积分项最大限制

public:
    /**
     * 构造函数
     * @param p 比例系数
     * @param i 积分系数
     * @param d 微分系数
     * @param min 最小输出
     * @param max 最大输出
     */
    PID(double p = 1.0, double i = 0.1, double d = 0.01,
        double min = -5.0, double max = 5.0,
        double windup = 10.0)
        : Kp(p), Ki(i), Kd(d),
        integral(0.0),
        previous_error(0.0),
        previous_input(0.0),
        output_min(min),
        output_max(max),
        windup_guard(windup) {
    }

    /**
     * 计算PID控制输出
     * @param setpoint 期望值
     * @param measured_value 实际测量值
     * @param dt 时间步长
     * @return 控制输出
     */
    double compute(double setpoint, double measured_value, double dt) {
        // 计算误差
        double error = setpoint - measured_value;

        // 比例项
        double proportional = Kp * error;

        // 积分项
        integral += error * dt;

        // 抗积分饱和
        integral = std::max(std::min(integral, windup_guard), -windup_guard);

        // 微分项（带滤波）
        double derivative = 0.0;
        if (dt > 0) {
            derivative = Kd * (error - previous_error) / dt;
        }

        // 计算输出
        double output = proportional + Ki * integral + derivative;

        // 输出限幅
        output = std::max(std::min(output, output_max), output_min);

        // 更新状态
        previous_error = error;
        previous_input = output;

        return output;
    }

    /**
     * 重置PID控制器状态
     */
    void reset() {
        integral = 0.0;
        previous_error = 0.0;
        previous_input = 0.0;
    }

    /**
     * 动态调整PID参数
     * @param p 比例系数
     * @param i 积分系数
     * @param d 微分系数
     */
    void setParameters(double p, double i, double d) {
        Kp = p;
        Ki = i;
        Kd = d;
    }
};

/**
 * 车辆模型类
 * 包含车辆的物理参数和状态变量，以及更新车辆状态的方法
 */
class Vehicle {
public:
    // 状态变量
    double x;      // 纵向位置 (m)
    double y;      // 横向位置 (m)
    double vx;     // 纵向速度 (m/s)
    double vy;     // 横向速度 (m/s)
    double psi;    // 航向角 (rad)
    double r;      // 横摆角速度 (rad/s)

    // 车辆参数
    double m;      // 质量 (kg)
    double Iz;     // 转动惯量 (kg*m^2)
    double lf;     // 前轴到质心距离 (m)
    double lr;     // 后轴到质心距离 (m)
    double width;  // 轮距 (m)
    double Caf;    // 前轮侧偏刚度 (N/rad)
    double Car;    // 后轮侧偏刚度 (N/rad)

    // 状态变化率限制
    double max_vy_rate;     // 最大横向加速度 (m/s²)
    double max_r_rate;      // 最大角加速度 (rad/s²)
    double max_delta_rate;  // 最大转向角变化率 (rad/s)
    double max_delta;       // 最大转向角 (rad)

    // 上一步的控制输入
    double prev_delta_f;

    /**
     * 构造函数，更精确地初始化车辆状态和参数
     */
    Vehicle()
        : x(0.0), y(0.0), vx(0.0), vy(0.0), psi(0.0), r(0.0),
        m(1500.0),      // 车辆质量
        Iz(2500.0),     // 转动惯量
        lf(1.2),        // 前轴到质心距离
        lr(1.3),        // 后轴到质心距离
        width(1.8),     // 轮距
        Caf(80000.0),   // 前轮侧偏刚度
        Car(80000.0),   // 后轮侧偏刚度
        max_vy_rate(3.0),   // 更大的横向加速度限制
        max_r_rate(0.8),    // 更大的角加速度限制
        max_delta_rate(0.5),// 更大的转向角变化率
        max_delta(0.7),     // 更大的最大转向角
        prev_delta_f(0.0) {
    }

    /**
     * 创建车辆的副本用于MPC预测
     */
    Vehicle clone() const {
        return *this;
    }

    /**
     * 更新车辆的纵向动力学状态
     * @param a 纵向加速度 (m/s^2)
     * @param dt 时间步长 (s)
     */
    void updateLongitudinal(double a, double dt) {
        // 限制加速度范围
        a = std::max(std::min(a, 3.0), -5.0);

        // 更新纵向速度
        vx += a * dt;
        vx = std::max(std::min(vx, 25.0), 1.0);  // 更宽松的速度范围
    }

    /**
     * 更新车辆的横向动力学状态
     * @param delta_f 前轮转向角 (rad)
     * @param dt 时间步长 (s)
     */
    void updateLateral(double delta_f, double dt) {
        // 限制转向角
        delta_f = std::max(std::min(delta_f, max_delta), -max_delta);

        // 限制转向角变化率
        double delta_change = delta_f - prev_delta_f;
        if (std::abs(delta_change) > max_delta_rate * dt) {
            delta_f = prev_delta_f +
                (delta_change > 0 ? 1 : -1) * max_delta_rate * dt;
        }
        prev_delta_f = delta_f;

        // 防止除零
        double safe_vx = std::max(vx, 1.0);

        // 计算侧偏角
        double alpha_f = delta_f - std::atan2(vy + lf * r, safe_vx);
        double alpha_r = -std::atan2(vy - lr * r, safe_vx);

        // 轮胎侧向力（线性模型）
        double Fyf = Caf * alpha_f;
        double Fyr = Car * alpha_r;

        // 更新横向速度和横摆角速度
        double ay = (Fyf + Fyr) / m - r * safe_vx;
        double ar = (lf * Fyf - lr * Fyr) / Iz;

        // 更新速度和角速度
        vy += ay * dt;
        r += ar * dt;

        // 限制横向速度和角速度
        vy = std::max(std::min(vy, 5.0), -5.0);
        r = std::max(std::min(r, 1.0), -1.0);

        // 更新航向角和位置
        psi += r * dt;

        // 确保航向角在-π到π之间
        while (psi > M_PI) psi -= 2 * M_PI;
        while (psi < -M_PI) psi += 2 * M_PI;

        // 更新位置（考虑速度和航向角）
        x += vx * std::cos(psi) * dt - vy * std::sin(psi) * dt;
        y += vx * std::sin(psi) * dt + vy * std::cos(psi) * dt;
    }

    /**
     * 获取车辆的状态误差
     * @param y_ref 横向位置参考值
     * @param psi_ref 航向角参考值
     * @return 状态误差向量
     */
    Vector4d getState(double y_ref, double psi_ref) const {
        double e_y = y - y_ref;
        double e_psi = psi - psi_ref;

        // 归一化航向角误差
        while (e_psi > M_PI) e_psi -= 2 * M_PI;
        while (e_psi < -M_PI) e_psi += 2 * M_PI;

        return Vector4d(e_y, e_psi, vy, r);
    }
};

/**
 * 生成平滑的S型周期轨迹，考虑车辆动力学约束
 */
 /**
  * ReferenceTrajectoryGenerator - 简化版
  * 生成满足车辆动力学约束的S型参考轨迹
  */
  /**
   * ReferenceTrajectoryGenerator - 简化版
   * 生成满足车辆动力学约束的S型参考轨迹
   */
class ReferenceTrajectoryGenerator {
private:
    // 轨迹参数
    double max_lateral_offset;   // 最大横向偏移
    double path_period;          // 轨迹周期长度
    double smoothness_factor;    // 平滑度因子

    // 车辆参数
    double vehicle_wheelbase;    // 车辆轴距
    double max_steering_angle;   // 最大转向角
    double max_lateral_accel;    // 最大横向加速度
    double max_steering_rate;    // 最大转向角速率

    bool trajectory_generated;

public:
    // 预生成的轨迹点 - 修改为公共可访问
    struct ReferencePoint {
        double time;    // 时间
        double x;       // x位置
        double y;       // y位置
        double psi;     // 航向角
        double speed;   // 速度
        double kappa;   // 曲率
    };
    vector<ReferencePoint> trajectory;

    /**
     * 构造函数
     * @param max_offset 最大横向偏移
     * @param period 轨迹周期
     * @param smoothness S曲线平滑度
     * @param wheelbase 车辆轴距
     * @param max_accel 最大横向加速度
     * @param max_delta 最大转向角
     * @param max_delta_rate 最大转向角速率
     */
    ReferenceTrajectoryGenerator(
        double max_offset = 0.5,    // 最大横向偏移
        double period = 200.0,      // 轨迹周期长度
        double smoothness = 5.0,    // 平滑度因子
        double wheelbase = 2.5,     // 轴距
        double max_accel = 2.0,     // 最大横向加速度
        double max_delta = 0.5,     // 最大转向角(rad)
        double max_delta_rate = 0.1 // 最大转向角速率(rad/s)
    ) : max_lateral_offset(max_offset),
        path_period(period),
        smoothness_factor(smoothness),
        vehicle_wheelbase(wheelbase),
        max_lateral_accel(max_accel),
        max_steering_angle(max_delta),
        max_steering_rate(max_delta_rate),
        trajectory_generated(false) {
    }

    /**
 * 根据索引获取参考点
 * @param index 轨迹点索引
 * @return 对应索引的参考点
 */
    const ReferencePoint& getPointAtIndex(size_t index) const {
        // 边界检查
        if (index >= trajectory.size()) {
            index = trajectory.size() - 1;
        }

        return trajectory[index];
    }

    /**
     * 获取从指定索引开始的N个参考点
     * @param start_index 起始索引
     * @param n_points 预测点数量
     * @return 未来参考点数组
     */
    std::vector<const ReferencePoint*> getFuturePointsFromIndex(size_t start_index, int n_points) const {
        std::vector<const ReferencePoint*> future_points;

        for (int i = 0; i < n_points; i++) {
            size_t idx = start_index + i;

            // 边界检查
            if (idx >= trajectory.size()) {
                idx = trajectory.size() - 1;
            }

            future_points.push_back(&trajectory[idx]);
        }

        return future_points;
    }

    /**
     * 生成平滑的S型周期轨迹
     * @param speed 基准速度
     * @param duration 轨迹持续时间
     * @param dt 时间步长
     * @return 是否成功生成轨迹
     */
    bool generateTrajectory(double speed, double duration, double dt = 0.01) {
        // 清空旧轨迹
        trajectory.clear();

        // 检查参数有效性
        if (speed <= 0 || duration <= 0 || dt <= 0) {
            std::cerr << "轨迹生成参数无效" << std::endl;
            return false;
        }

        // 基于曲率和最大横向加速度约束计算安全速度和偏移量
        // 初步估计最大曲率 - 使用正弦函数，最大曲率出现在波峰和波谷
        double estimated_max_curvature = max_lateral_offset * pow(2 * M_PI / path_period, 2);

        // 计算安全的最大速度
        double max_safe_speed = sqrt(max_lateral_accel / estimated_max_curvature);

        // 如果给定速度超过安全速度，调整速度
        double actual_offset = max_lateral_offset;
        double actual_speed = speed;

        if (speed > max_safe_speed) {
            // 降低速度而不是减小偏移量
            actual_speed = max_safe_speed * 0.9;  // 留10%的安全余量
            std::cout << "警告: 速度过高可能导致横向加速度超限，已调整速度为 "
                << actual_speed << "m/s (原始值: " << speed << "m/s)" << std::endl;
        }

        // 检查转向角速率约束
        double max_kappa_rate = max_steering_rate / (vehicle_wheelbase * actual_speed);
        double required_kappa_rate = 2 * M_PI * actual_speed * actual_offset / (path_period * path_period);

        if (required_kappa_rate > max_kappa_rate) {
            // 如果预计超出转向角速率约束，可以再次调整速度或者增加周期长度
            double adjusted_period = path_period * sqrt(required_kappa_rate / max_kappa_rate);
            std::cout << "警告: 转向角速率约束可能超限，建议增加轨迹周期至 "
                << adjusted_period << "m (当前: " << path_period << "m)" << std::endl;
        }

        // 生成轨迹点，使用正弦函数生成S形轨迹 - 与Python代码保持一致
        for (double t = 0; t <= duration; t += dt) {
            double x = actual_speed * t;

            // 计算归一化阶段 (0 到 2π)
            double phase = 2 * M_PI * x / path_period;

            // 使用正弦函数生成S形轨迹
            double y = actual_offset * sin(phase);

            // 计算航向角 (psi) - 使用解析导数
            double dy_dx = actual_offset * 2 * M_PI / path_period * cos(phase);
            double psi = atan2(dy_dx, 1.0);

            // 计算曲率 (kappa) - 二阶导数
            double d2y_dx2 = -actual_offset * pow(2 * M_PI / path_period, 2) * sin(phase);
            double kappa = d2y_dx2 / pow(1 + dy_dx * dy_dx, 1.5);

            // 存储轨迹点
            trajectory.push_back({
                t,          // 时间
                x,          // x位置
                y,          // y位置
                psi,        // 航向角
                actual_speed, // 速度
                kappa       // 曲率
                });
        }

        // 验证轨迹满足动力学约束
        validateTrajectory();

        trajectory_generated = true;

        std::cout << "已生成 " << trajectory.size() << " 个轨迹点，总长度 "
            << trajectory.back().x << "m，持续时间 "
            << trajectory.back().time << "s" << std::endl;

        return true;
    }

    /**
     * 验证轨迹是否满足动力学约束
     * @return 是否满足约束
     */
    bool validateTrajectory() {
        if (trajectory.empty()) return false;

        bool valid = true;
        int violations = 0;

        for (size_t i = 1; i < trajectory.size(); ++i) {
            const auto& point = trajectory[i];
            const auto& prev = trajectory[i - 1];

            // 检查横向加速度约束 (规则1)
            double lateral_accel = pow(point.speed, 2) * fabs(point.kappa);
            if (lateral_accel > max_lateral_accel) {
                std::cerr << "约束违反: 横向加速度 " << lateral_accel
                    << " > " << max_lateral_accel << " at x=" << point.x << std::endl;
                violations++;
                valid = false;
            }

            // 检查转向角约束 (规则2)
            double steering = atan(vehicle_wheelbase * fabs(point.kappa));
            if (fabs(steering) > max_steering_angle) {
                std::cerr << "约束违反: 转向角 " << fabs(steering)
                    << " > " << max_steering_angle << " at x=" << point.x << std::endl;
                violations++;
                valid = false;
            }

            // 检查转向角变化率约束 (规则3)
            double dt = point.time - prev.time;
            if (dt > 0) {
                double dkappa = fabs(point.kappa - prev.kappa);
                double steering_rate = vehicle_wheelbase * dkappa * point.speed / dt;
                if (steering_rate > max_steering_rate) {
                    std::cerr << "约束违反: 转向角变化率 " << steering_rate
                        << " > " << max_steering_rate << " at x=" << point.x << std::endl;
                    violations++;
                    valid = false;
                }
            }
        }

        if (!valid) {
            std::cerr << "警告: 共有 " << violations << " 个约束违反。"
                << "考虑减小速度、增加轨迹周期长度或减小横向偏移量。" << std::endl;
        }
        else {
            std::cout << "轨迹验证通过：满足所有动力学约束！" << std::endl;
        }

        return valid;
    }

    /**
     * 获取轨迹是否已生成
     * @return 轨迹生成状态
     */
    bool isTrajectoryGenerated() const {
        return trajectory_generated;
    }

    /**
     * 获取车辆轴距
     * @return 车辆轴距
     */
    double getWheelbase() const {
        return vehicle_wheelbase;
    }

    /**
     * 计算最大转向角
     * @param speed 速度
     * @return 最大转向角
     */
    double computeMaxSteeringAngle(double speed) const {
        return max_steering_angle;
    }

    /**
     * 保存生成的轨迹到CSV文件
     * @param filename 文件名
     * @return 是否成功保存
     */
    bool saveTrajectoryToFile(const std::string& filename) const {
        if (!trajectory_generated || trajectory.empty()) {
            std::cerr << "错误: 轨迹未生成或为空" << std::endl;
            return false;
        }

        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "错误: 无法打开文件 " << filename << std::endl;
            return false;
        }

        // 写入表头
        file << "time,x,y,psi,speed,curvature" << std::endl;

        // 写入数据
        for (const auto& point : trajectory) {
            file << point.time << ","
                << point.x << ","
                << point.y << ","
                << point.psi << ","
                << point.speed << ","
                << point.kappa << std::endl;
        }

        file.close();
        std::cout << "轨迹已保存到 " << filename << std::endl;
        return true;
    }
};
#endif // VEHICLE_CONTROL_BASE_H