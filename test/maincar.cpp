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

using namespace speedbot;

int main(int argc, char** argv)
{
    if (argc < 3)
        return 0;

    // cv::Mat rgb = cv::imread(argv[1]);
    // cv::Mat depth = cv::imread(argv[2], -1);

    speedbot::CloudMatch cloud_match = speedbot::CloudMatch();

    Euler euler_convertor = speedbot::Euler();
    
    /* cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 3565.5546875000, 0., 1030.6973876953, 0., 3565.6948242188, 765.4660034180, 0., 0., 1.);
    cv::Mat distCoeff = (cv::Mat_<double>(5, 1) << -0.0677342713, 0.1611212343, -0.0003548863, 0.0001102431, -0.3424454033);

    cloud_match.setCameraMatrix(cameraMatrix);
    cloud_match.setDistCoeff(distCoeff);

    cv::Rect roi_rect(950.0, 150.0, 800.0, 400.0);

    cloud_match.setDownSampleSize(0.5);
    
    // cloud_match.extractTemplatePointCloud(rgb, depth, roi_rect);
    // cloud_match.saveTemplatePC("./template_pc.ply");

    // draw template results
    // Eigen::Matrix4f _trans;
    // _trans.setIdentity();
    // auto img = cloud_match.drawCloudResultImg(rgb, _trans);
    // cv::imwrite("./result.png", img);

    cloud_match.loadTemplatePC("./template_pc.ply");

    cloud_match.setDownSampleSize(0.5);
    cloud_match.calCloudMatchingPose(rgb, depth, roi_rect);
    auto _trans = cloud_match.getMatchingResult();

    std::cout << " _trans : " << _trans << std::endl;

    auto img = cloud_match.drawCloudResultImg(rgb, _trans);

    cv::imwrite("./match_result.png", img); */

    pcl::PointCloud<pcl::PointXYZ>::Ptr source_pc(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_pc(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::io::loadPLYFile(argv[1], *source_pc);
    pcl::io::loadPLYFile(argv[2], *target_pc);

    pcl::PointCloud<pcl::PointXYZ>::Ptr downsample_source_pc(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsample_target_pc(new pcl::PointCloud<pcl::PointXYZ>());

    downsample_source_pc = cloud_match.downSampleCloud(source_pc, 0.5);
    downsample_target_pc = cloud_match.downSampleCloud(target_pc, 0.5);

    float extrinsic_quat[7] = { -4.6670269959754830e+02, -2.8953427598971319e+02,
       -4.5442781088065158e+02, -3.2516388072478608e-03,
       -4.3804113474932380e-03, 3.9115717249047395e-03,
       9.9997746897024375e-01 };

    Eigen::Quaternionf extrinsic_q(extrinsic_quat[6], extrinsic_quat[3], extrinsic_quat[4], extrinsic_quat[5]);
    Eigen::Matrix3f extrinsic_m = extrinsic_q.toRotationMatrix();

    Eigen::Matrix4f correct_extrinsic;
    correct_extrinsic.setIdentity();
    correct_extrinsic.block<3, 3>(0, 0) = extrinsic_m;
    correct_extrinsic(0, 3) = extrinsic_quat[0];
    correct_extrinsic(1, 3) = extrinsic_quat[1];
    correct_extrinsic(2, 3) = extrinsic_quat[2];

    std::cout << "correct_extrinsic : " << correct_extrinsic.inverse() << std::endl;

    float source_hand_pos[6] = { -519.35,-259.77,-497.95,100.64,88.02,-162.89 }; // 模板数据位置
    float target_hand_pos[6] = { -652.48,-217.41,-336.5,154.37,88.69,-110.99 };  // 运动后的数据位置

    Eigen::Matrix4f source_hand_m;
    source_hand_m.setIdentity();
    auto tmp = euler_convertor.Euler2mat<float>(source_hand_pos[3] * M_PI / 180.0, source_hand_pos[4] * M_PI / 180.0, source_hand_pos[5] * M_PI / 180.0, "rzyx");
    std::cout << "tmp : " << tmp << std::endl;
    source_hand_m.block<3, 3>(0, 0) = tmp;
    source_hand_m(0, 3) = source_hand_pos[0];
    source_hand_m(1, 3) = source_hand_pos[1];
    source_hand_m(2, 3) = source_hand_pos[2];

    std::cout << "source_hand_m : " << source_hand_m << std::endl;

    Eigen::Matrix4f target_hand_m;
    target_hand_m.setIdentity();
    target_hand_m.block<3, 3>(0, 0) = euler_convertor.Euler2mat<float>(target_hand_pos[3] * M_PI / 180.0, target_hand_pos[4] * M_PI / 180.0, target_hand_pos[5] * M_PI / 180.0, "rzyx");
    target_hand_m(0, 3) = target_hand_pos[0];
    target_hand_m(1, 3) = target_hand_pos[1];
    target_hand_m(2, 3) = target_hand_pos[2];

    std::cout << "target_hand_m : " << target_hand_m << std::endl;

    Eigen::Quaternionf temp_quat(target_hand_m.block<3, 3>(0, 0));
    //std::cout << "temp_quat : " << temp_quat << std::endl;
    std::cout << "temp_quat: (" << temp_quat.w() << ", " << temp_quat.x() << ", " << temp_quat.y() << ", " << temp_quat.z() << ")" << std::endl;


    float point_source_hand_pos[6] = { -568.64, 40.94, -205.15, 100.68, 88.02,-162.85 }; // 指点数据
    Eigen::Matrix4f point_source_hand_m;
    point_source_hand_m.setIdentity();
    point_source_hand_m.block<3, 3>(0, 0) = euler_convertor.Euler2mat<float>(point_source_hand_pos[3] * M_PI / 180.0, point_source_hand_pos[4] * M_PI / 180.0, point_source_hand_pos[5] * M_PI / 180.0, "rzyx");
    point_source_hand_m(0, 3) = point_source_hand_pos[0];
    point_source_hand_m(1, 3) = point_source_hand_pos[1];
    point_source_hand_m(2, 3) = point_source_hand_pos[2];


    // 固定相机外参
    /*correct_extrinsic << -9.786374639842800732e-03, 2.509768679234025690e-03, -9.999489626638878859e-01, -4.208721569869099994e+02,
        1.349103173030309633e-02, 9.999061650756795316e-01, 2.377626232198393197e-03, 2.215900329822460208e+03,
        9.998610998205004208e-01, -1.346707484292036447e-02, -9.819315705006366121e-03, 4.458853405832839599e+03,
        0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00; */

    pcl::PointCloud<pcl::PointXYZ>::Ptr trans_downsample_source_pc(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr trans_downsample_target_pc(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*downsample_source_pc, *trans_downsample_source_pc, source_hand_m * correct_extrinsic);
    pcl::transformPointCloud(*downsample_target_pc, *trans_downsample_target_pc, target_hand_m * correct_extrinsic);

    pcl::io::savePLYFile("./trans_downsample_source_pc.ply", *trans_downsample_source_pc);
    pcl::io::savePLYFile("./trans_downsample_target_pc.ply", *trans_downsample_target_pc);

    Eigen::Matrix4f _trans = cloud_match.matchPointClouds(trans_downsample_source_pc, trans_downsample_target_pc);

    float mse = cloud_match.getFinalConvergenceMse();
    std::cout << "mse : " << mse << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr _trans_downsample_source_pc(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*trans_downsample_source_pc, *_trans_downsample_source_pc, _trans);
    pcl::io::savePLYFile("./_trans_downsample_source_pc.ply", *_trans_downsample_source_pc);

    std::cout << " _trans : " << _trans << std::endl;

    Eigen::Matrix4f cal_target_hand = _trans;
    Eigen::Matrix3f cal_target_hand_m = cal_target_hand.block<3, 3>(0, 0);
    auto euler_angle = euler_convertor.Mat2euler<float>(cal_target_hand_m, "rzyx");
    std::cout << "cal_target_hand : " << euler_angle.transpose() * 180.0 / M_PI << std::endl;
    std::cout << "cal_target_hand trans : " << (cal_target_hand.block(0, 3, 3, 1)).norm() << std::endl;

    Eigen::Matrix4f cal_point_hand_pos = _trans * point_source_hand_m;
    Eigen::Matrix3f cal_point_hand_pos_m = cal_point_hand_pos.block<3, 3>(0, 0);
    Eigen::Quaternionf cal_quat(cal_point_hand_pos_m);
    //std::cout << "cal_quat : " << cal_quat << std::endl;
    std::cout << "cal_quat coefficients: (" << cal_quat.w() << ", " << cal_quat.x() << ", " << cal_quat.y() << ", " << cal_quat.z() << ")" << std::endl;

    euler_angle = euler_convertor.Mat2euler<float>(cal_point_hand_pos_m, "rzyx");
    std::cout << "euler_angle : " << euler_angle.transpose() * 180.0 / M_PI << std::endl;
    std::cout << "x, y, z : " << cal_point_hand_pos.block(0, 3, 3, 1) << std::endl;

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
