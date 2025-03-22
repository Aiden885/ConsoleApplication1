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

    // ��ʼ����ǿ�͹켣������
    ReferenceTrajectoryGenerator trajectory_generator(
        0.0,    // ������ƫ��
        200.0,  // �켣���ڳ���
        5.0,    // ƽ��������
        2.5,    // ���
        2.0,    // ��������ٶ�
        0.5,    // ���ת���
        0.1     // ���ת�������
    );

    // �������߹켣
    double sim_time = 20.0;  // ģ��ʱ�䣨�룩
    double base_speed = 5.0; // ��׼�ٶȣ�m/s��
    double dt_traj = 0.01;   // �켣����ʱ�䲽��

    // ���ɹ켣������
    trajectory_generator.generateTrajectory(base_speed, sim_time, dt_traj);
    trajectory_generator.saveTrajectoryToFile("generated_trajectory.csv");

    // ��ʼ��MPC������ - ����Ԥ�ⲽ�����������
    ModelPredictiveController mpc(
        10,     // Ԥ�ⲽ�� - ��30���ٵ�15
        0.05    // Ԥ�ⲽ��
    );

    // ����MPC������Ȩ������߸�������
    mpc.setWeights(
        2000.0,  // λ�����Ȩ�� - ����
        5000.0,  // ��������Ȩ�� - ����
        50.0,    // �����ٶ����Ȩ��
        30.0,    // ��ڽ��ٶ����Ȩ��
        0.1,     // ת�������Ȩ��
        0.2      // ת��Ǳ仯��Ȩ��
    );

    // ��ʼ��PID�������������ٶȿ��ƣ�
    PID speed_pid(20, 0.02, 0.02, -2.0, 2.0);


    // �������
    double dt = 0.01;         // ʱ�䲽��
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

    // ��ѭ��ǰ����ʼ������λ�õ��켣���
    if (!trajectory_generator.trajectory.empty()) {
        const auto& start_point = trajectory_generator.getPointAtIndex(0);
        vehicle.x = start_point.x;
        vehicle.y = start_point.y;
        vehicle.psi = start_point.psi;
        vehicle.vx = base_speed * 0.0; // ��ʼ�ٶ���ΪĿ���ٶȵ�һ��
    }

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

        double speed_error = vehicle.vx - target_speed;

        // �������ͳ��
        total_lateral_error += std::abs(lateral_error);
        max_lateral_error = std::max(max_lateral_error, std::abs(lateral_error));
        total_heading_error += std::abs(heading_error);
        max_heading_error = std::max(max_heading_error, std::abs(heading_error));
        total_speed_error += std::abs(speed_error);
        max_speed_error = std::max(max_speed_error, std::abs(speed_error));

        // MPC����������ת��� - ���뵱ǰ����
        double mpc_delta_f = mpc.compute(vehicle, trajectory_generator, prev_delta_f, trajectory_index);



        // �����������
        double max_speed_reduction = 2.0;  // ����ٶȽ��� 2 m/s
        double error_threshold = 0.1;      // ���������ֵ

        // ���ں�����̬����Ŀ���ٶ�
        double speed_reduction = std::min(max_speed_reduction, std::abs(lateral_error) / error_threshold);
        double adjusted_target_speed = target_speed - speed_reduction;

        // ������ƣ�ʹ��PID������ٶ�
        double a = speed_pid.compute(adjusted_target_speed, vehicle.vx, dt);

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
            << target_speed << ","
            << lateral_error << ","
            << heading_error << ","
            << speed_error << ","
            << mpc_delta_f << ","
            << a << std::endl;

        // ÿ�����һ��״̬
        if (i % 100 == 0) {
            std::cout << "Time: " << std::fixed << std::setprecision(1) << t
                << "s, y: " << std::setprecision(2) << vehicle.y
                << "m, y_ref: " << y_ref
                << "m, err: " << lateral_error
                << "m, ��(MPC): " << mpc_delta_f
                << "rad, v: " << vehicle.vx << "m/s" << std::endl;
        }

        // ���¹켣����������ʵ���ٶȺ͹켣����ȷ������
        // �������ȼ��켣�㣬����dt_trajΪʱ��������
        double distance_traveled = vehicle.vx * dt;
        double distance_per_point = base_speed * dt_traj; // �켣��֮��ľ���
        int index_increment = static_cast<int>(std::round(distance_traveled / distance_per_point));

        // ȷ������ǰ��һ����
        trajectory_index += std::max(1, index_increment);
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