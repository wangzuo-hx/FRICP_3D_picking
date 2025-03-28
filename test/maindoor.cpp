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

//int main(int argc, char** argv)
int main()
{
    /*if (argc < 3)
        return 0;*/

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

    pcl::io::loadPLYFile("..\\..\\data\\current1\\door.ply", *source_pc);
    pcl::io::loadPLYFile("..\\..\\data\\current1\\door_test11.ply", *target_pc);

    pcl::PointCloud<pcl::PointXYZ>::Ptr downsample_source_pc(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsample_target_pc(new pcl::PointCloud<pcl::PointXYZ>());

    downsample_source_pc = cloud_match.downSampleCloud(source_pc, 1.0);
    downsample_target_pc = cloud_match.downSampleCloud(target_pc, 1.0);

    float extrinsic_quat[7] = { -4257.91,-2160.1,-388.753,-0.7049747,0.00327764, 0.70921013,0.00457388 };

    Eigen::Quaternionf extrinsic_q(extrinsic_quat[6], extrinsic_quat[3], extrinsic_quat[4], extrinsic_quat[5]);
    Eigen::Matrix3f extrinsic_m = extrinsic_q.toRotationMatrix();

    Eigen::Matrix4f correct_extrinsic;
    correct_extrinsic.setIdentity();
    correct_extrinsic.block<3, 3>(0, 0) = extrinsic_m;
    correct_extrinsic(0, 3) = extrinsic_quat[0];
    correct_extrinsic(1, 3) = extrinsic_quat[1];
    correct_extrinsic(2, 3) = extrinsic_quat[2];

    std::cout << "correct_extrinsic : " << correct_extrinsic << std::endl;

    // 固定相机外参
    /*correct_extrinsic << -9.786374639842800732e-03, 2.509768679234025690e-03, -9.999489626638878859e-01, -4.208721569869099994e+02,
        1.349103173030309633e-02, 9.999061650756795316e-01, 2.377626232198393197e-03, 2.215900329822460208e+03,
        9.998610998205004208e-01, -1.346707484292036447e-02, -9.819315705006366121e-03, 4.458853405832839599e+03,
        0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00; */

    pcl::PointCloud<pcl::PointXYZ>::Ptr trans_downsample_source_pc(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr trans_downsample_target_pc(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*downsample_source_pc, *trans_downsample_source_pc, correct_extrinsic);
    pcl::transformPointCloud(*downsample_target_pc, *trans_downsample_target_pc, correct_extrinsic);

    pcl::io::savePLYFile("./trans_downsample_source_pc.ply", *trans_downsample_source_pc);
    pcl::io::savePLYFile("./trans_downsample_target_pc.ply", *trans_downsample_target_pc);

    Eigen::Matrix4f _trans = cloud_match.matchPointClouds(trans_downsample_source_pc, trans_downsample_target_pc);

    float mse = cloud_match.getFinalConvergenceMse();
    std::cout << "mse : " << mse << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr _trans_downsample_source_pc(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*trans_downsample_source_pc, *_trans_downsample_source_pc, _trans);
    pcl::io::savePLYFile("./_trans_downsample_source_pc.ply", *_trans_downsample_source_pc);

    std::cout << " _trans : " << _trans << std::endl;

    float source_hand_pos[6] = { -3406.92, -1693.76, -187.89, -90.77, -2.51, -176.62 };
    //float target_hand_pos[6] = { -3430.75, -1674.51, -188.12, -91.34, -2.37, -178.32 };
    float target_hand_pos[6] = { -3410.77, -1732.02, -178.80, -93.53, -2.29, -178.51 };


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
    target_hand_m.block<3, 3>(0, 0) = euler_convertor.Euler2mat<float>(target_hand_pos[3] * M_PI / 180.0, target_hand_pos[4] * M_PI / 180.0 , target_hand_pos[5] * M_PI / 180.0, "rzyx");
    target_hand_m(0, 3) = target_hand_pos[0];
    target_hand_m(1, 3) = target_hand_pos[1];
    target_hand_m(2, 3) = target_hand_pos[2];

    std::cout << "target_hand_m : " << target_hand_m << std::endl;

    /*if (std::string(argv[3]) == "train")
    {
        Eigen::Matrix4f cal_hand_trans = target_hand_m * source_hand_m.inverse();
        Eigen::Matrix4f trans_compensate = cal_hand_trans * _trans.inverse();

        cv::Mat trans_compensate_mat;
        cv::eigen2cv(trans_compensate, trans_compensate_mat);

        cv::FileStorage file_compensate("file_compensate.yaml", cv::FileStorage::WRITE);
        file_compensate << "compensate" << trans_compensate_mat;
        file_compensate.release();
    }
    else if (std::string(argv[3]) == "test")
    {
        cv::FileStorage file_compensate("file_compensate.yaml", cv::FileStorage::READ);
        cv::Mat compensate_mat;
        file_compensate["compensate"] >> compensate_mat;
        file_compensate.release();

        Eigen::Matrix4f hand_compensate_m;
        cv::cv2eigen(compensate_mat, hand_compensate_m);

        _trans = hand_compensate_m * _trans;
    }*/

    Eigen::Matrix4f cal_target_hand = _trans * source_hand_m;
    Eigen::Matrix3f cal_target_hand_m = cal_target_hand.block<3, 3>(0, 0);
    auto euler_angle = euler_convertor.Mat2euler<float>(cal_target_hand_m, "rzyx");
    std::cout << "cal_target_hand : " << euler_angle.transpose() * 180.0 / M_PI << std::endl;
    std::cout << "cal_target_hand translate : " << cal_target_hand.block(0, 3, 3, 1).transpose() << std::endl;
    std::cout << "cal_target_hand trans : " << (cal_target_hand.block(0, 3, 3, 1) - target_hand_m.block(0, 3, 3, 1)).norm() << std::endl;

    Eigen::Matrix4f result = (target_hand_m.inverse()) * _trans * source_hand_m;
    std::cout << "result : " << result << std::endl;

    Eigen::Matrix4f cal_trans = target_hand_m * source_hand_m.inverse();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cal_trans_downsample_source_pc(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*trans_downsample_source_pc, *cal_trans_downsample_source_pc, cal_trans);
    pcl::io::savePLYFile("./cal_trans_downsample_source_pc.ply", *cal_trans_downsample_source_pc);

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
