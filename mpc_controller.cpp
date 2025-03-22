#include "vehicle_control_base.h"
#include <qpOASES.hpp>
#include <iomanip>
#include <algorithm>

/**
 * 改进的模型预测控制器（MPC）
 * 针对直接访问轨迹生成器的版本
 */
class ModelPredictiveController {
private:
    // MPC参数
    int N;              // 预测步数
    double dt;          // 预测步长
    double max_delta;   // 最大转向角
    double max_delta_rate; // 最大转向角变化率

    // 控制约束
    double delta_min;   // 最小转向角
    double delta_max;   // 最大转向角
    double ddelta_min;  // 最小转向角变化率
    double ddelta_max;  // 最大转向角变化率

    // QP问题规模
    int n_states;       // 状态变量数量
    int n_controls;     // 控制变量数量
    int n_variables;    // 总变量数量 (N*n_states + (N-1)*n_controls)
    int n_constraints;  // 约束数量

    // 代价函数权重矩阵
    Eigen::MatrixXd Q;  // 状态误差权重
    Eigen::MatrixXd R;  // 控制输入权重
    Eigen::MatrixXd R_delta; // 控制变化率权重

    // 车辆模型参数
    double m;      // 质量
    double Iz;     // 转动惯量
    double lf;     // 前轴到质心距离
    double lr;     // 后轴到质心距离
    double Caf;    // 前轮侧偏刚度
    double Car;    // 后轮侧偏刚度

public:
    /**
     * 构造函数
     * @param prediction_horizon 预测步数
     * @param step_size 预测步长
     */
    ModelPredictiveController(
        int prediction_horizon = 10,
        double step_size = 0.1)
        : N(prediction_horizon),
        dt(step_size),
        max_delta(0.8),
        max_delta_rate(0.2) {

        // 初始化问题规模
        n_states = 4;    // [e_y, e_psi, vy, r]
        n_controls = 1;  // [delta]
        n_variables = N * n_states + (N - 1) * n_controls;
        n_constraints = (N - 1) * n_states     // 状态方程约束
            + 2 * (N - 1) * n_controls; // 控制约束和变化率约束

        // 初始化控制约束
        delta_min = -max_delta;
        delta_max = max_delta;
        ddelta_min = -max_delta_rate;
        ddelta_max = max_delta_rate;

        // 初始化代价函数权重矩阵 - 默认值，可通过setter方法调整
        Q = Eigen::MatrixXd::Zero(n_states, n_states);
        Q(0, 0) = 500.0;  // 横向位置误差
        Q(1, 1) = 1000.0; // 航向角误差
        Q(2, 2) = 50.0;   // 横向速度误差
        Q(3, 3) = 30.0;   // 横摆角速度误差

        R = Eigen::MatrixXd::Zero(n_controls, n_controls);
        R(0, 0) = 1.0;   // 转向角输入

        R_delta = Eigen::MatrixXd::Zero(n_controls, n_controls);
        R_delta(0, 0) = 2.0;  // 转向角变化率

        // 从车辆中获取参数
        Vehicle vehicle;
        m = vehicle.m;
        Iz = vehicle.Iz;
        lf = vehicle.lf;
        lr = vehicle.lr;
        Caf = vehicle.Caf;
        Car = vehicle.Car;
    }

    /**
     * 计算最优控制输入
     * @param vehicle 当前车辆状态
     * @param trajectory_generator 轨迹生成器
     * @param prev_delta 上一步转向角
     * @return 最优转向角
     */
     /**
      * 计算最优控制输入 - 使用索引而非最近点查找
      */
    double compute(const Vehicle& vehicle,
        const ReferenceTrajectoryGenerator& trajectory_generator,
        double prev_delta,
        size_t current_index) {  // 使用轨迹索引代替时间
        // 检查轨迹是否已生成
        if (!trajectory_generator.isTrajectoryGenerated() || trajectory_generator.trajectory.empty()) {
            std::cerr << "错误: 轨迹未生成或为空" << std::endl;
            return prev_delta;
        }

        // 直接获取当前索引对应的参考点
        const auto& ref_point = trajectory_generator.getPointAtIndex(current_index);

        // 计算当前状态误差
        Eigen::Vector4d current_state = computeStateError(vehicle, ref_point);

        // 建立状态空间模型（线性化）
        Eigen::MatrixXd A, B;
        linearizeModel(vehicle.vx, A, B);

        // 从轨迹中获取未来N个参考点
        auto future_refs = trajectory_generator.getFuturePointsFromIndex(current_index, N);

        // 设置QP问题
        Eigen::MatrixXd H;
        Eigen::VectorXd g;
        Eigen::MatrixXd A_constraints;
        Eigen::VectorXd lb, ub;

        setupQPMatrices(A, B, current_state, future_refs, prev_delta,
            vehicle.vx, H, g, A_constraints, lb, ub);

        // 求解QP问题并返回结果
        return solveQP(H, g, A_constraints, lb, ub, prev_delta);
    }

    /**
     * 设置权重矩阵
     * @param position_weight 位置误差权重
     * @param heading_weight 航向角误差权重
     * @param velocity_weight 速度误差权重
     * @param yaw_rate_weight 横摆角速度误差权重
     * @param steering_weight 转向角输入权重
     * @param steering_rate_weight 转向角变化率权重
     */
    void setWeights(double position_weight, double heading_weight,
        double velocity_weight, double yaw_rate_weight,
        double steering_weight, double steering_rate_weight) {
        Q(0, 0) = position_weight;
        Q(1, 1) = heading_weight;
        Q(2, 2) = velocity_weight;
        Q(3, 3) = yaw_rate_weight;

        R(0, 0) = steering_weight;

        R_delta(0, 0) = steering_rate_weight;
    }

    /**
     * 设置转向约束
     * @param max_steering_angle 最大转向角
     * @param max_steering_rate 最大转向角变化率
     */
    void setSteeringConstraints(double max_steering_angle, double max_steering_rate) {
        max_delta = max_steering_angle;
        max_delta_rate = max_steering_rate;

        delta_min = -max_delta;
        delta_max = max_delta;
        ddelta_min = -max_delta_rate;
        ddelta_max = max_delta_rate;
    }


    /**
     * 计算车辆相对于参考点的状态误差
     * @param vehicle 车辆状态
     * @param ref_point 参考点
     * @return 状态误差向量 [e_y, e_psi, vy, r]
     */
    Eigen::Vector4d computeStateError(const Vehicle& vehicle,
        const ReferenceTrajectoryGenerator::ReferencePoint& ref_point) {
        Eigen::Vector4d error;

        // 计算全局坐标系下的横向误差
        double dx = vehicle.x - ref_point.x;
        double dy = vehicle.y - ref_point.y;

        // 将误差转换到轨迹坐标系
        double cos_ref = cos(ref_point.psi);
        double sin_ref = sin(ref_point.psi);

        // 横向误差 (e_y) - 确保符号正确
        error(0) = dy * cos_ref - dx * sin_ref;

        // 航向角误差 (e_psi)
        double heading_error = vehicle.psi - ref_point.psi;
        // 归一化到 [-π, π]
        while (heading_error > M_PI) heading_error -= 2 * M_PI;
        while (heading_error < -M_PI) heading_error += 2 * M_PI;
        error(1) = heading_error;

        // 横向速度 (vy)
        error(2) = vehicle.vy;

        // 横摆角速度 (r)
        error(3) = vehicle.r;

        return error;
    }

    /**
     * 线性化车辆模型，得到状态空间矩阵A和B
     * @param vx 纵向速度
     * @param A 输出状态转移矩阵
     * @param B 输出控制输入矩阵
     */
    void linearizeModel(double vx, Eigen::MatrixXd& A, Eigen::MatrixXd& B) {
        // 确保vx不为零
        double safe_vx = std::max(vx, 0.1);

        // 初始化状态空间矩阵
        A = Eigen::MatrixXd::Identity(n_states, n_states);
        B = Eigen::MatrixXd::Zero(n_states, n_controls);

        // 计算矩阵元素
        double a11 = 0.0;  // d(e_y)/d(e_y)
        double a12 = vx;   // d(e_y)/d(e_psi)
        double a13 = 1.0;  // d(e_y)/d(vy)
        double a14 = 0.0;  // d(e_y)/d(r)

        double a21 = 0.0;  // d(e_psi)/d(e_y)
        double a22 = 0.0;  // d(e_psi)/d(e_psi)
        double a23 = 0.0;  // d(e_psi)/d(vy)
        double a24 = 1.0;  // d(e_psi)/d(r)

        double a31 = 0.0;  // d(vy)/d(e_y)
        double a32 = 0.0;  // d(vy)/d(e_psi)
        double a33 = -(Caf + Car) / (m * safe_vx);  // d(vy)/d(vy)
        double a34 = -safe_vx - (lf * Caf - lr * Car) / (m * safe_vx);  // d(vy)/d(r)

        double a41 = 0.0;  // d(r)/d(e_y)
        double a42 = 0.0;  // d(r)/d(e_psi)
        double a43 = -(lf * Caf - lr * Car) / (Iz * safe_vx);  // d(r)/d(vy)
        double a44 = -(lf * lf * Caf + lr * lr * Car) / (Iz * safe_vx);  // d(r)/d(r)

        // 计算输入矩阵元素
        double b11 = 0.0;  // d(e_y)/d(delta)
        double b21 = 0.0;  // d(e_psi)/d(delta)
        double b31 = Caf / m;  // d(vy)/d(delta)
        double b41 = lf * Caf / Iz;  // d(r)/d(delta)

        // 构建连续时间状态空间矩阵
        Eigen::MatrixXd Ac = Eigen::MatrixXd::Zero(n_states, n_states);
        Eigen::MatrixXd Bc = Eigen::MatrixXd::Zero(n_states, n_controls);

        Ac << a11, a12, a13, a14,
            a21, a22, a23, a24,
            a31, a32, a33, a34,
            a41, a42, a43, a44;

        Bc << b11, b21, b31, b41;

        // 使用前向欧拉法离散化
        A = Eigen::MatrixXd::Identity(n_states, n_states) + dt * Ac;
        B = dt * Bc;
    }

    /**
     * 设置QP问题矩阵
     */
    void setupQPMatrices(
        const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
        const Eigen::Vector4d& current_state,
        const vector<const ReferenceTrajectoryGenerator::ReferencePoint*>& future_refs,
        double prev_delta,
        double current_velocity,
        Eigen::MatrixXd& H, Eigen::VectorXd& g,
        Eigen::MatrixXd& A_constraints, Eigen::VectorXd& lb, Eigen::VectorXd& ub) {

        // 确定预测长度
        int horizon = std::min(static_cast<int>(future_refs.size()), N);

        // 明确定义问题规模
        n_variables = horizon * n_states + (horizon - 1) * n_controls;
        n_constraints = (horizon - 1) * n_states + 2 * (horizon - 1) * n_controls;

        // 重置矩阵尺寸
        H.resize(n_variables, n_variables);
        g.resize(n_variables);
        A_constraints.resize(n_constraints, n_variables);
        lb.resize(n_constraints);
        ub.resize(n_constraints);

        H.setZero();
        g.setZero();
        A_constraints.setZero();
        lb.setZero();
        ub.setZero();

        // 正则化
        double regularization = 1e-4;

        // 速度因子 - 高速下增加航向权重
        double speed_factor = std::min(current_velocity / 10.0, 1.0);

        // 动态权重
        Eigen::MatrixXd Q_dynamic = Eigen::MatrixXd::Zero(n_states, n_states);
        Q_dynamic(0, 0) = Q(0, 0) * (1.0 + speed_factor);     // 横向位置
        Q_dynamic(1, 1) = Q(1, 1) * (1.0 + 0.5 * speed_factor); // 航向角
        Q_dynamic(2, 2) = Q(2, 2);  // 横向速度
        Q_dynamic(3, 3) = Q(3, 3);  // 横摆角速度

        Eigen::MatrixXd R_dynamic(1, 1);
        R_dynamic << R(0, 0) * (1.0 + 0.5 * speed_factor);

        Eigen::MatrixXd R_delta_dynamic(1, 1);
        R_delta_dynamic << R_delta(0, 0) * (1.0 + 0.5 * speed_factor);

        // 构建 Hessian 矩阵
        for (int i = 0; i < horizon; i++) {
            int state_idx = i * n_states;
            // 状态代价
            H.block(state_idx, state_idx, n_states, n_states) =
                Q_dynamic + regularization * Eigen::MatrixXd::Identity(n_states, n_states);

            // 计算参考状态
            Eigen::Vector4d x_ref;
            double vx_ref = future_refs[i]->speed;
            double kappa_ref = future_refs[i]->kappa;
            x_ref(0) = 0.0; // e_y_ref（误差定义中已包含参考值）
            x_ref(1) = 0.0; // e_psi_ref（误差定义中已包含参考值）
            x_ref(2) = vx_ref * kappa_ref; // v_y_ref
            x_ref(3) = vx_ref * kappa_ref; // r_ref

            // 线性项 g
            g.segment(state_idx, n_states) = -2 * Q_dynamic * x_ref;

            // 控制输入代价
            if (i < horizon - 1) {
                int control_idx = horizon * n_states + i * n_controls;
                H.block(control_idx, control_idx, n_controls, n_controls) =
                    R_dynamic + regularization * Eigen::MatrixXd::Identity(n_controls, n_controls);

                // 控制变化率
                if (i > 0) {
                    int prev_control_idx = horizon * n_states + (i - 1) * n_controls;
                    H.block(control_idx, prev_control_idx, n_controls, n_controls) = -R_delta_dynamic;
                    H.block(prev_control_idx, control_idx, n_controls, n_controls) = -R_delta_dynamic;
                    H.block(control_idx, control_idx, n_controls, n_controls) +=
                        2 * R_delta_dynamic + regularization * Eigen::MatrixXd::Identity(n_controls, n_controls);
                }
            }
        }


        // 约束构建
        int constraint_idx = 0;

        // 动态约束
        for (int i = 0; i < horizon - 1; i++) {
            int x_idx = i * n_states;
            int u_idx = horizon * n_states + i * n_controls;
            int next_x_idx = (i + 1) * n_states;

            // 状态转移约束: x_{t+1} = Ax_t + Bu_t
            A_constraints.block(constraint_idx, x_idx, n_states, n_states) = -A;
            A_constraints.block(constraint_idx, u_idx, n_states, n_controls) = -B;
            A_constraints.block(constraint_idx, next_x_idx, n_states, n_states) =
                Eigen::MatrixXd::Identity(n_states, n_states);

            // 初始状态
            if (i == 0) {
                lb.segment(constraint_idx, n_states) = A * current_state;
                ub.segment(constraint_idx, n_states) = A * current_state;
            }
            else {
                lb.segment(constraint_idx, n_states).setZero();
                ub.segment(constraint_idx, n_states).setZero();
            }
            constraint_idx += n_states;
        }

        // 控制约束
        double delta_range = max_delta;
        double ddelta_range = max_delta_rate * dt; // 转换为离散时间下的限制

        for (int i = 0; i < horizon - 1; i++) {
            int u_idx = horizon * n_states + i * n_controls;

            // 转向角约束
            A_constraints.block(constraint_idx, u_idx, n_controls, n_controls) =
                Eigen::MatrixXd::Identity(n_controls, n_controls);

            lb.segment(constraint_idx, n_controls) =
                Eigen::VectorXd::Constant(n_controls, -delta_range);
            ub.segment(constraint_idx, n_controls) =
                Eigen::VectorXd::Constant(n_controls, delta_range);

            constraint_idx += n_controls;

            // 转向角变化率约束
            if (i > 0) {
                int prev_u_idx = horizon * n_states + (i - 1) * n_controls;
                A_constraints.block(constraint_idx, u_idx, n_controls, n_controls) =
                    Eigen::MatrixXd::Identity(n_controls, n_controls);
                A_constraints.block(constraint_idx, prev_u_idx, n_controls, n_controls) =
                    -Eigen::MatrixXd::Identity(n_controls, n_controls);

                lb.segment(constraint_idx, n_controls) =
                    Eigen::VectorXd::Constant(n_controls, -ddelta_range);
                ub.segment(constraint_idx, n_controls) =
                    Eigen::VectorXd::Constant(n_controls, ddelta_range);
            }
            else {
                // 第一步控制输入的变化率
                A_constraints.block(constraint_idx, u_idx, n_controls, n_controls) =
                    Eigen::MatrixXd::Identity(n_controls, n_controls);

                lb.segment(constraint_idx, n_controls) =
                    Eigen::VectorXd::Constant(n_controls, prev_delta - ddelta_range);
                ub.segment(constraint_idx, n_controls) =
                    Eigen::VectorXd::Constant(n_controls, prev_delta + ddelta_range);
            }

            constraint_idx += n_controls;
        }
    }

    /**
     * 求解QP问题
     */
    double solveQP(
        const Eigen::MatrixXd& H,
        const Eigen::VectorXd& g,
        const Eigen::MatrixXd& A_constraints,
        const Eigen::VectorXd& lb,
        const Eigen::VectorXd& ub,
        double prev_delta) {

        // 初始化QP求解器
        qpOASES::QProblem qp(n_variables, n_constraints);
        qpOASES::Options options;
        options.printLevel = qpOASES::PL_NONE;  // 关闭求解器输出
        options.enableRegularisation = qpOASES::BT_TRUE;  // 启用正则化
        options.numRegularisationSteps = 3;  // 增加正则化步数
        qp.setOptions(options);

        // 求解QP问题
        int nWSR = 2000;  // 最大迭代次数

        // 将Eigen矩阵转换为qpOASES可接受的格式
        double* H_data = const_cast<double*>(H.data());
        double* g_data = const_cast<double*>(g.data());
        double* A_data = const_cast<double*>(A_constraints.data());
        double* lb_data = const_cast<double*>(lb.data());
        double* ub_data = const_cast<double*>(ub.data());

        // 解决QP问题
        qpOASES::returnValue ret = qp.init(H_data, g_data, A_data, nullptr, nullptr, lb_data, ub_data, nWSR);


        // 获取优化结果
        Eigen::VectorXd solution(n_variables);
        qp.getPrimalSolution(solution.data());

        // 提取第一步控制输入
        double best_delta = solution[n_states];  // 状态变量后面是第一个控制输入

        // 安全性检查
        best_delta = std::max(std::min(best_delta, max_delta), -max_delta);

        // 平滑控制变化
        double delta_change = best_delta - prev_delta;
        double max_delta_change = max_delta_rate * dt; // 每个时间步的最大变化
        if (std::abs(delta_change) > max_delta_change) {
            best_delta = prev_delta + (delta_change > 0 ? max_delta_change : -max_delta_change);
        }

        return best_delta;
    }
};