// FRICP_test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "FRICP/CloudMatch.h"
#include "euler.h"

#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using namespace speedbot;

int main()
{
    std::string command = std::string("test");

    speedbot::CloudMatch cloud_match = speedbot::CloudMatch(); // 点云配准对象
    speedbot::CloudMatch car_match = speedbot::CloudMatch();
    Euler euler_convertor = speedbot::Euler(); // 欧拉角转换对象

    std::cout << "command : " << command << std::endl;

    if (command == std::string("train"))
    {
        std::cout << "no train operations ! " << std::endl;
    }
    else if (command == std::string("test"))
    {
        std::cout << "begin test " << std::endl;

        float downsample_size = std::atof("0.5");

        float door_robot_assemble_template_handpose[6] = { -443.67,-866.38,49.97,-179.80,-12.59,171.30 };  // 装配准确时的机器人工具末端位置 Ft

        Eigen::Matrix4f Ft;
        Ft.setIdentity();
        Ft.block<3, 3>(0, 0) =  euler_convertor.Euler2mat<float>(door_robot_assemble_template_handpose[3] * M_PI / 180.0, \
            door_robot_assemble_template_handpose[4] * M_PI / 180.0, door_robot_assemble_template_handpose[5] * M_PI / 180.0, "rzyx");
        Ft(0, 3) = door_robot_assemble_template_handpose[0];
        Ft(1, 3) = door_robot_assemble_template_handpose[1];
        Ft(2, 3) = door_robot_assemble_template_handpose[2];


        float door_robot_correct_template_handpose[6] = { -3406.92,-1693.76,-187.89,-90.77,-2.51,-176.62 }; // 车门侧纠偏时候的模板姿态 Gt

        Eigen::Matrix4f Gt;
        Gt.setIdentity();
        Gt.block<3, 3>(0, 0) = euler_convertor.Euler2mat<float>(door_robot_correct_template_handpose[3] * M_PI / 180.0, \
            door_robot_correct_template_handpose[4] * M_PI / 180.0, door_robot_correct_template_handpose[5] * M_PI / 180.0, "rzyx");
        Gt(0, 3) = door_robot_correct_template_handpose[0];
        Gt(1, 3) = door_robot_correct_template_handpose[1];
        Gt(2, 3) = door_robot_correct_template_handpose[2];

        std::cout << " ---- Gt : " << Gt << std::endl;

        float door_robot_correct_curr_handpose[6] = { -3406.92,-1693.76,-187.89,-90.77,-2.51,-176.62 }; // 车门纠偏时候的实时姿态 Gc
        Eigen::Matrix4f Gc;
        Gc.setIdentity();
        Gc.block<3, 3>(0, 0) = euler_convertor.Euler2mat<float>(door_robot_correct_curr_handpose[3] * M_PI / 180.0, \
            door_robot_correct_curr_handpose[4] * M_PI / 180.0, door_robot_correct_curr_handpose[5] * M_PI / 180.0, "rzyx");
        Gc(0, 3) = door_robot_correct_curr_handpose[0];
        Gc(1, 3) = door_robot_correct_curr_handpose[1];
        Gc(2, 3) = door_robot_correct_curr_handpose[2];

        std::cout << " ---- Gc : " << Gc << std::endl;

        // 录入外参
        float extrinsic_quat[7] = { -4257.91,-2160.1,-388.753,-0.7049747,0.00327764, 0.70921013,0.00457388 };  //车门纠偏外参

        Eigen::Quaternionf extrinsic_q(extrinsic_quat[6], extrinsic_quat[3], extrinsic_quat[4], extrinsic_quat[5]);
        Eigen::Matrix3f extrinsic_m = extrinsic_q.toRotationMatrix();

        Eigen::Matrix4f correct_extrinsic;
        correct_extrinsic.setIdentity();
        correct_extrinsic.block<3, 3>(0, 0) = extrinsic_m;
        correct_extrinsic(0, 3) = extrinsic_quat[0];
        correct_extrinsic(1, 3) = extrinsic_quat[1];
        correct_extrinsic(2, 3) = extrinsic_quat[2];

        std::cout << " ------ correct_camera_extrinsic : " << correct_extrinsic << std::endl;


        pcl::PointCloud<pcl::PointXYZ>::Ptr door_correct_template_camera_pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr door_correct_template_base_pc(new pcl::PointCloud<pcl::PointXYZ>()); // Ot
        pcl::io::loadPLYFile("C:\\coding\\FRICP-master\\data\\template\\door.ply", *door_correct_template_camera_pc);
        door_correct_template_camera_pc = cloud_match.downSampleCloud(door_correct_template_camera_pc, downsample_size);

        // 获取基于Base的车门点云数据 Ot
        pcl::transformPointCloud(*door_correct_template_camera_pc, *door_correct_template_base_pc, correct_extrinsic);

        // 载入基于Base的实施车门点云数据 Oc
        pcl::PointCloud<pcl::PointXYZ>::Ptr door_correct_curr_camera_pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr door_correct_curr_base_pc(new pcl::PointCloud<pcl::PointXYZ>()); // Oc
        pcl::io::loadPLYFile("C:\\coding\\FRICP-master\\data\\current\\door.ply", *door_correct_curr_camera_pc);
        door_correct_curr_camera_pc = cloud_match.downSampleCloud(door_correct_curr_camera_pc, downsample_size);

        pcl::transformPointCloud(*door_correct_curr_camera_pc, *door_correct_curr_base_pc, correct_extrinsic); // Oc under base

        Eigen::Matrix4f To = cloud_match.matchPointClouds(door_correct_template_camera_pc, door_correct_curr_camera_pc); // To
        To = correct_extrinsic * To * correct_extrinsic.inverse();   

        std::cout << " ---- finish To calculation : " << To << std::endl;

        pcl::PointCloud<pcl::PointXYZ>::Ptr trans_door_pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*door_correct_template_base_pc, *trans_door_pc, To);
        pcl::io::savePLYFile(".\\trans_door_pc.ply", *trans_door_pc);
        pcl::io::savePLYFile(".\\origin_door_pc.ply", *door_correct_curr_base_pc);


        // -------------计算 Ct 与 Cc ------------------
        
        // ----- 获得车身侧的眼在手上外参 -----
        float car_extrinsic_quat[7] = { -4.6670269959754830e+02, -2.8953427598971319e+02,
       -4.5442781088065158e+02, -3.2516388072478608e-03,
       -4.3804113474932380e-03, 3.9115717249047395e-03,
       9.9997746897024375e-01 };

        Eigen::Quaternionf car_extrinsic_q(car_extrinsic_quat[6], car_extrinsic_quat[3], car_extrinsic_quat[4], car_extrinsic_quat[5]);
        Eigen::Matrix3f car_extrinsic_m = car_extrinsic_q.toRotationMatrix();

        Eigen::Matrix4f car_extrinsic;
        car_extrinsic.setIdentity();
        car_extrinsic.block<3, 3>(0, 0) = car_extrinsic_m;
        car_extrinsic(0, 3) = car_extrinsic_quat[0];
        car_extrinsic(1, 3) = car_extrinsic_quat[1];
        car_extrinsic(2, 3) = car_extrinsic_quat[2];

        // ------ 获取模板车身位置的工具末端数据 --------
        float car_template_handpose[6] = { -1800.04, -201.64, -806.41, 141.72, 89.20, -139.95 }; // 模板
        Eigen::Matrix4f car_template_handpose_matrix;
        car_template_handpose_matrix.setIdentity();
        car_template_handpose_matrix.block<3, 3>(0, 0) = euler_convertor.Euler2mat<float>(car_template_handpose[3] * M_PI / 180.0, \
            car_template_handpose[4] * M_PI / 180.0, car_template_handpose[5] * M_PI / 180.0, "rzyx");
        car_template_handpose_matrix(0, 3) = car_template_handpose[0];
        car_template_handpose_matrix(1, 3) = car_template_handpose[1];
        car_template_handpose_matrix(2, 3) = car_template_handpose[2];

        // 载入基于Base的模板车身点云数据 Ct
        Eigen::Matrix4f car_trans_base = car_template_handpose_matrix * car_extrinsic;
        pcl::PointCloud<pcl::PointXYZ>::Ptr car_template_camera_pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr car_template_base_pc(new pcl::PointCloud<pcl::PointXYZ>()); // Ct
        pcl::io::loadPLYFile("C:\\coding\\FRICP-master\\data\\template\\car.ply", *car_template_camera_pc);
        car_template_camera_pc = car_match.downSampleCloud(car_template_camera_pc, downsample_size);

        pcl::transformPointCloud(*car_template_camera_pc, *car_template_base_pc, car_trans_base); // Ct under base

        // ------ 获取实时车身位置的工具末端数据 --------
        float car_curr_handpose[6] = { -1800.12,-214.54,-776.22,141.94,89.20,-140.60 }; // 实时
        Eigen::Matrix4f car_curr_handpose_matrix;
        car_curr_handpose_matrix.setIdentity();
        car_curr_handpose_matrix.block<3, 3>(0, 0) = euler_convertor.Euler2mat<float>(car_curr_handpose[3] * M_PI / 180.0, \
            car_curr_handpose[4] * M_PI / 180.0, car_curr_handpose[5] * M_PI / 180.0, "rzyx");
        car_curr_handpose_matrix(0, 3) = car_curr_handpose[0];
        car_curr_handpose_matrix(1, 3) = car_curr_handpose[1];
        car_curr_handpose_matrix(2, 3) = car_curr_handpose[2];

        // 载入基于Base的实时车身点云数据 Cc
        car_trans_base = car_curr_handpose_matrix * car_extrinsic;
        pcl::PointCloud<pcl::PointXYZ>::Ptr car_curr_camera_pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr car_curr_base_pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::io::loadPLYFile("C:\\coding\\FRICP-master\\data\\current\\car.ply", *car_curr_camera_pc);
        car_curr_camera_pc = car_match.downSampleCloud(car_curr_camera_pc, downsample_size);

        pcl::transformPointCloud(*car_curr_camera_pc, *car_curr_base_pc, car_trans_base); // Cc under base

        // 计算Tc
        Eigen::Matrix4f Tc = car_match.matchPointClouds(car_template_camera_pc, car_curr_camera_pc);

        Eigen::Matrix3f Tc_m = Tc.block<3, 3>(0, 0);
        auto tm_euler_angle = euler_convertor.Mat2euler<float>(Tc_m, "rzyx");
        std::cout << "euler_angle : " << tm_euler_angle.transpose() * 180.0 / M_PI << std::endl;
        std::cout << "x, y, z : " << Tc.block(0, 3, 3, 1) << std::endl;

        Tc = car_curr_handpose_matrix * car_extrinsic * Tc * car_extrinsic.inverse() * car_template_handpose_matrix.inverse();

        pcl::PointCloud<pcl::PointXYZ>::Ptr trans_car_pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*car_template_base_pc, *trans_car_pc, Tc);
        pcl::io::savePLYFile(".\\trans_car_pc.ply", *trans_car_pc);
        pcl::io::savePLYFile(".\\template_car_pc.ply", *car_template_base_pc);
        pcl::io::savePLYFile(".\\origin_car_pc.ply", *car_curr_base_pc);

        std::cout << " ----- finish Tc calculation : " << Tc << std::endl;

        // F_c = T_c * F_t * G_t^-1 * T_o^-1 * G_c
        Eigen::Matrix4f Fc = Tc * Ft * Gt.inverse() * To.inverse() * Gc;

        std::cout << " ----- finish Fc calculation : " << Fc << std::endl;

        Eigen::Matrix3f Fc_m = Fc.block<3, 3>(0, 0);
        Eigen::Quaternionf cal_quat(Fc_m);
        //std::cout << "cal_quat : " << cal_quat << std::endl;
        std::cout << "cal_quat coefficients: (" << cal_quat.w() << ", " << cal_quat.x() << ", " << cal_quat.y() << ", " << cal_quat.z() << ")" << std::endl;

        auto euler_angle = euler_convertor.Mat2euler<float>(Fc_m, "rzyx");
        std::cout << "euler_angle : " << euler_angle.transpose() * 180.0 / M_PI << std::endl;
        std::cout << "x, y, z : " << Fc.block(0, 3, 3, 1) << std::endl;
    }

    return 1;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
