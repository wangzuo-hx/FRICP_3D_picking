// Copyright (c) Speedbot Corporation. All rights reserved.
// Licensed under the MIT License.

/*
 * @Author       : Lintao Zheng
 * @Email        : lintaozheng1991@gmail.com
 * @Github       : https://github.com/zlt1991
 * @Date         : 2020-10-04 20:04:41
 * @LastEditTime : 2021-05-05 02:29:50
 */

#ifndef EULER_H
#define EULER_H

// System headers
#include <iostream>
#include <string>
#include <unordered_map>

// Library headers
#include "Eigen/Dense"
#include <Eigen/Geometry> 

#include <Eigen/Core>

// Project headers

namespace speedbot {

/**
 * @brief 欧拉角、四元数、旋转矩阵转换类
 *
 */
class Euler {
public:
    Euler() = default;
    ~Euler() = default;
    template <typename T>
    static Eigen::Matrix<T, 3, 3> Euler2mat(T ai, T aj, T ak, std::string axes = "sxyz");
    template <typename T>
    static Eigen::Matrix<T, 3, 1> Mat2euler(Eigen::Matrix<T, 3, 3>& M, std::string axes = "sxyz");
    template <typename T>
    static Eigen::Quaternion<T> Euler2quat(T ai, T aj, T ak, std::string axes = "sxyz");
    template <typename T>
    static Eigen::Matrix<T, 3, 1> Quat2euler(Eigen::Quaternion<T> q, std::string axes = "sxyz");

private:
    // static const float _FLOAT_EPS;
    // static const float _EPS4;
    // static const float _FLOAT_EPS;
    // static const float _EPS4;
    /**
     * @brief 将旋转方式转化为内部vector表示, s开头表示静态轴（固定轴，外旋），r开头表示动态轴（旋转轴，内旋）
     *
     * @param axes
     * @return Eigen::Vector4i
     */
    static Eigen::Vector4i Axes2vector(std::string axes) {
        Eigen::Vector4i result;
        const std::unordered_map<std::string, std::function<void(Eigen::Vector4i&)>> AXESMAP{
            {"sxyz", [](Eigen::Vector4i& result) { result<<0, 0, 0, 0; }},
            {"sxyx", [](Eigen::Vector4i& result) { result<<0, 0, 1, 0; }},
            {"sxzy", [](Eigen::Vector4i& result) { result<<0, 1, 0, 0; }},
            {"sxzx", [](Eigen::Vector4i& result) { result<<0, 1, 1, 0; }},
            {"syzx", [](Eigen::Vector4i& result) { result<<1, 0, 0, 0; }},
            {"syzy", [](Eigen::Vector4i& result) { result<<1, 0, 1, 0; }},
            {"syxz", [](Eigen::Vector4i& result) { result<<1, 1, 0, 0; }},
            {"syxy", [](Eigen::Vector4i& result) { result<<1, 1, 1, 0; }},
            {"szxy", [](Eigen::Vector4i& result) { result<<2, 0, 0, 0; }},
            {"szxz", [](Eigen::Vector4i& result) { result<<2, 0, 1, 0; }},
            {"szyx", [](Eigen::Vector4i& result) { result<<2, 1, 0, 0; }},
            {"szyz", [](Eigen::Vector4i& result) { result<<2, 1, 1, 0; }},
            {"rzyx", [](Eigen::Vector4i& result) { result<<0, 0, 0, 1; }},
            {"rxyx", [](Eigen::Vector4i& result) { result<<0, 0, 1, 1; }},
            {"ryzx", [](Eigen::Vector4i& result) { result<<0, 1, 0, 1; }},
            {"rxzx", [](Eigen::Vector4i& result) { result<<0, 1, 1, 1; }},
            {"rxzy", [](Eigen::Vector4i& result) { result<<1, 0, 0, 1; }},
            {"ryzy", [](Eigen::Vector4i& result) { result<<1, 0, 1, 1; }},
            {"rzxy", [](Eigen::Vector4i& result) { result<<1, 1, 0, 1; }},
            {"ryxy", [](Eigen::Vector4i& result) { result<<1, 1, 1, 1; }},
            {"ryxz", [](Eigen::Vector4i& result) { result<<2, 0, 0, 1; }},
            {"rzxz", [](Eigen::Vector4i& result) { result<<2, 0, 1, 1; }},
            {"rxyz", [](Eigen::Vector4i& result) { result<<2, 1, 0, 1; }},
            {"rzyz", [](Eigen::Vector4i& result) { result<<2, 1, 1, 1; }},
        };
        const auto end = AXESMAP.end();
        auto it = AXESMAP.find(axes);
        if (it != end) {
            it->second(result);
        } else {
            result << -1, -1, -1, -1;
            std::cerr << "no match axes string!!!!" << std::endl;
        }
        return result;
    }
};

// const float Euler::_FLOAT_EPS = 1e-7;
// const float Euler::_EPS4 = 4e-7;

/**
 * @brief 欧拉角转旋转矩阵(注意这里的欧拉角单位为弧度)
 *
 * @tparam T 数据类型
 * @param ai 根据欧拉角旋转顺序的第一个旋转角
 * @param aj 根据欧拉角旋转顺序的第二个旋转角
 * @param ak 根据欧拉角旋转顺序的第三个旋转角
 * @param axes 设定的欧拉角旋转顺序，如sxyz,rxyz...
 * @return Eigen::Matrix<T,3,3>  3x3旋转矩阵
 */
template <typename T>
Eigen::Matrix<T, 3, 3> Euler::Euler2mat(T ai, T aj, T ak, std::string axes) {
    Eigen::Vector4i next_axis;
    next_axis << 1, 2, 0, 1;
    Eigen::Vector4i axisVec = Axes2vector(axes); // firstaxis, parity, repetition, frame
    int firstaxis = axisVec(0);
    int parity = axisVec(1);
    int repetition = axisVec(2);
    int frame = axisVec(3);
    int i = firstaxis;
    int j = next_axis(i + parity);
    int k = next_axis(i - parity + 1);

    if (frame != 0) {
        float tmp = ai;
        ai = ak;
        ak = tmp;
    }
    if (parity != 0) {
        ai = -ai;
        aj = -aj;
        ak = -ak;
    }

    T si, sj, sk;
    si = sin(ai);
    sj = sin(aj);
    sk = sin(ak);
    T ci, cj, ck;
    ci = cos(ai);
    cj = cos(aj);
    ck = cos(ak);
    T cc, cs;
    cc = ci * ck;
    cs = ci * sk;
    T sc, ss;
    sc = si * ck;
    ss = si * sk;

    Eigen::Matrix<T, 3, 3> M = Eigen::Matrix<T, 3, 3>::Identity(3, 3);
    if (repetition != 0) {
        M(i, i) = cj;
        M(i, j) = sj * si;
        M(i, k) = sj * ci;
        M(j, i) = sj * sk;
        M(j, j) = -cj * ss + cc;
        M(j, k) = -cj * cs - sc;
        M(k, i) = -sj * ck;
        M(k, j) = cj * sc + cs;
        M(k, k) = cj * cc - ss;
    } else {
        M(i, i) = cj * ck;
        M(i, j) = sj * sc - cs;
        M(i, k) = sj * cc + ss;
        M(j, i) = cj * sk;
        M(j, j) = sj * ss + cc;
        M(j, k) = sj * cs - sc;
        M(k, i) = -sj;
        M(k, j) = cj * si;
        M(k, k) = cj * ci;
    }
    return M;
}

/**
 * @brief  旋转矩阵转欧拉角(注意这里的欧拉角单位为弧度)
 *
 * @tparam T  数据类型
 * @param M  旋转矩阵
 * @param axes  设定的欧拉角旋转顺序
 * @return Eigen::Matrix<T,3,1> 对应的欧拉角向量
 */
template <typename T>
Eigen::Matrix<T, 3, 1> Euler::Mat2euler(Eigen::Matrix<T, 3, 3>& M, std::string axes) {
    Eigen::Vector4i next_axis;
    next_axis << 1, 2, 0, 1;
    Eigen::Vector4i axisVec = Axes2vector(axes); // firstaxis, parity, repetition, frame
    int firstaxis = axisVec(0);
    int parity = axisVec(1);
    int repetition = axisVec(2);
    int frame = axisVec(3);

    int i = firstaxis;
    int j = next_axis(i + parity);
    int k = next_axis(i - parity + 1);

    T ax, ay, az;
    if (repetition != 0) {
        T sy = sqrt(M(i, j) * M(i, j) + M(i, k) * M(i, k));
        if (sy > 4e-7) {
            ax = atan2(M(i, j), M(i, k));
            ay = atan2(sy, M(i, i));
            az = atan2(M(j, i), -M(k, i));
        } else {
            ax = atan2(-M(j, k), M(j, j));
            ay = atan2(sy, M(i, i));
            az = 0.0;
        }
    } else {
        T cy = sqrt(M(i, i) * M(i, i) + M(j, i) * M(j, i));
        if (cy > 4e-7) {
            ax = atan2(M(k, j), M(k, k));
            ay = atan2(-M(k, i), cy);
            az = atan2(M(j, i), M(i, i));
        } else {
            ax = atan2(-M(j, k), M(j, j));
            ay = atan2(-M(k, i), cy);
            az = 0.0;
        }
    }

    if (parity != 0) {
        ax = -ax;
        ay = -ay;
        az = -az;
    }
    if (frame != 0) {
        T tmp = ax;
        ax = az;
        az = tmp;
    }
    Eigen::Matrix<T, 3, 1> res;
    res << ax, ay, az;
    return res;
}

/**
 * @brief 欧拉角转四元数(注意这里的欧拉角单位为弧度)
 *
 * @tparam T 数据类型
 * @param ai 根据欧拉角旋转顺序的第一个欧拉角
 * @param aj 根据欧拉角旋转顺序的第二个欧拉角
 * @param ak 根据欧拉角旋转顺序的第三个欧拉角
 * @param axes 设置的欧拉角旋转顺序
 * @return Eigen::Quaternion<T> 返回的四元数， w,x,y,z
 */

template <typename T>
Eigen::Quaternion<T> Euler::Euler2quat(T ai, T aj, T ak, std::string axes) {
    Eigen::Matrix<T, 3, 3> mat = Euler2mat(ai, aj, ak, axes);
    Eigen::Quaternion<T> q(mat);
    return q;
}

/**
 * @brief 四元数转欧拉角(注意这里的欧拉角单位为弧度)
 *
 * @tparam T 数据类型
 * @param q  四元数
 * @param axes  设置的欧拉角旋转顺序, 如sxyz，rxyz
 * @return Eigen::Matrix<T,3,1> 旋转矩阵
 */
template <typename T>
Eigen::Matrix<T, 3, 1> Euler::Quat2euler(Eigen::Quaternion<T> q, std::string axes) {
    Eigen::Matrix<T, 3, 3> rot = q.toRotationMatrix();
    return Mat2euler(rot, axes);
}

} // namespace namese speedbot

#endif