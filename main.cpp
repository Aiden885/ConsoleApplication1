#include "vehicle_control_base.h"
#include <qpOASES.hpp>
#include <iomanip>
#include <algorithm>

// Include the trajectory generator and MPC controller cpp
#include "mpc_controller.cpp"
/**
 * ������ - �������Ʒ���
 * ʹ����ǿ�͹켣�������͸Ľ���MPC������
 */
int main() {
    // ��ʼ������
    Vehicle vehicle;
    vehicle.vx = 0.0; // ��ʼ�ٶ���Ϊ0

    const double DT_BASE = 0.01;           // ����ʱ�䲽��(��)
    const double DT_CONTROL = DT_BASE;     // ���Ƹ��²���
    const double DT_TRAJ = DT_BASE;        // �켣���ɲ���
    const double DT_MPC = DT_BASE;         // MPCԤ�ⲽ��Ҳ��Ϊ0.01��
    const int MPC_HORIZON = 5;

    // ��ʼ��PID�������������ٶȿ��ƣ�
    PID speed_pid(20.0, 0.05, 0.02, -2.0, 2.0);

    // ��ʼ����ǿ�͹켣������ - ����PID�ͳ�������
    ReferenceTrajectoryGenerator trajectory_generator(
        speed_pid,      // �ٶ�PID������
        vehicle,        // ����ģ��
        1.0,            // ������ƫ��
        200.0,          // �켣���ڳ���
        5.0,            // ƽ��������
        2.5,            // ���
        2.0,            // ��������ٶ�
        0.5,            // ���ת���
        0.1             // ���ת�������
    );

    // �������߹켣 - ���ڻῼ�Ǽ��ٹ���
    double sim_time = 20.0;  // ģ��ʱ�䣨�룩
    double base_speed = 5.0; // Ŀ���ٶȣ�m/s��
    double dt_traj = DT_BASE;   // �켣����ʱ�䲽��
    

    // ���ɹ켣������
    trajectory_generator.generateTrajectory(base_speed, sim_time + 20.0, dt_traj);
    trajectory_generator.saveTrajectoryToFile("generated_trajectory.csv");

    // ��ʼ��MPC������
    ModelPredictiveController mpc(
        MPC_HORIZON,     // Ԥ�ⲽ��
        DT_MPC    // Ԥ�ⲽ��
    );

    // ��ǿMPC��������Ȩ��
    mpc.setWeights(
        0.0,   // λ�����Ȩ��
        0.0,   // ��������Ȩ��
        0.0,     // �����ٶ����Ȩ��
        0.0,      // ��ڽ��ٶ����Ȩ��
        50.0,       // ת�������Ȩ��
        50.0        // ת��Ǳ仯��Ȩ��
    );



    // �������
    double dt = DT_BASE;         // ʱ�䲽��
    int steps = static_cast<int>(sim_time / dt);

    // �ο��ٶ�
    double target_speed = base_speed;  // m/s

    // ͳ��ָ��
    double total_lateral_error = 0.0;
    double max_lateral_error = 0.0;
    double total_heading_error = 0.0;
    double max_heading_error = 0.0;
    double total_speed_error = 0.0;
    double max_speed_error = 0.0;

    // ��ʼ������ļ�
    std::ofstream data_file("vehicle_control_data.csv");
    data_file << "time,x,y,vx,vy,psi,"
        << "y_ref,psi_ref,speed_ref,"
        << "lateral_error,heading_error,speed_error,"
        << "mpc_delta,pid_delta,"
        << "acceleration" << std::endl;

    // ��һ����ת�������
    double prev_delta_f = 0.0;

    // ���ó���״̬��׼��ʵ�ʷ���
    vehicle.x = 0.0;
    vehicle.y = 0.0;
    vehicle.psi = 0.0;
    vehicle.vx = 0.0;
    vehicle.vy = 0.0;
    vehicle.r = 0.0;

    std::cout << "��ʼ����..." << std::endl;

    // ��ѭ��
    size_t trajectory_index = 0; // ��ǰ�켣����

    for (int i = 0; i < steps; ++i) {
        double t = i * dt;  // ��ǰ����ʱ��

        // ȷ�������������켣����
        if (trajectory_index >= trajectory_generator.trajectory.size()) {
            trajectory_index = trajectory_generator.trajectory.size() - 1;
        }

        // ֱ��ʹ��������ȡ�ο���
        const auto& ref_point = trajectory_generator.getPointAtIndex(trajectory_index);
        double y_ref = ref_point.y;
        double psi_ref = ref_point.psi;
        double speed_ref = ref_point.speed;

        // �������ָ��
        double lateral_error = vehicle.y - y_ref;
        double heading_error = vehicle.psi - psi_ref;
        while (heading_error > M_PI) heading_error -= 2 * M_PI;
        while (heading_error < -M_PI) heading_error += 2 * M_PI;

        double speed_error = vehicle.vx - speed_ref;

        // �������ͳ��
        total_lateral_error += std::abs(lateral_error);
        max_lateral_error = std::max(max_lateral_error, std::abs(lateral_error));
        total_heading_error += std::abs(heading_error);
        max_heading_error = std::max(max_heading_error, std::abs(heading_error));
        total_speed_error += std::abs(speed_error);
        max_speed_error = std::max(max_speed_error, std::abs(speed_error));

        // MPC����������ת��� - ���뵱ǰ����
        double mpc_delta_f = mpc.compute(vehicle, trajectory_generator, prev_delta_f, trajectory_index);


        // ������ƣ�ʹ��PID������ٶȣ��ο��ٶ�����ʹ�ù켣���е��ٶ�
        double a = speed_pid.compute(speed_ref, vehicle.vx, dt);

        // ���³���״̬
        vehicle.updateLongitudinal(a, dt);
        vehicle.updateLateral(mpc_delta_f, dt);  // ʹ��MPC��ת���

        // ������һ��ת���
        prev_delta_f = mpc_delta_f;

        // д�����ݵ�CSV�ļ�
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
                << "m, ��(MPC): " << mpc_delta_f
                << "rad, v: " << vehicle.vx << "m/s" << std::endl;
        }

        // ����xλ��Ѱ����һ�����ʵĹ켣��
        double min_dist = std::numeric_limits<double>::max();
        size_t closest_idx = trajectory_index;

        // ��δ��һС�ι켣�в�����ӽ��ĵ�
        for (size_t j = trajectory_index; j < std::min(trajectory_index + 50, trajectory_generator.trajectory.size()); ++j) {
            double dist = std::abs(trajectory_generator.trajectory[j].x - vehicle.x);
            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = j;
            }
        }

        // ���¹켣����
        trajectory_index = closest_idx + 1; // ǰ������һ����
    }

    // ���㲢���ͳ��ָ��
    std::cout << "\n��������ͳ��:" << std::endl;
    std::cout << "ƽ���������: " << total_lateral_error / steps << "m" << std::endl;
    std::cout << "���������: " << max_lateral_error << "m" << std::endl;
    std::cout << "ƽ����������: " << total_heading_error / steps << "rad" << std::endl;
    std::cout << "���������: " << max_heading_error << "rad" << std::endl;
    std::cout << "ƽ���ٶ����: " << total_speed_error / steps << "m/s" << std::endl;
    std::cout << "����ٶ����: " << max_speed_error << "m/s" << std::endl;

    data_file.close();

    std::cout << "\n������ɣ������ѱ��浽 vehicle_control_data.csv" << std::endl;
    std::cout << "�켣�����ѱ��浽 generated_trajectory.csv" << std::endl;

    return 0;
}