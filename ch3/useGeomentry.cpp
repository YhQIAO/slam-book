//
// Created by qyh on 2021/8/4.
//

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;
using namespace std;

int main() {

    Matrix3d rotation_matrix = Matrix3d::Identity();
    // cout << rotation_matrix << endl;
    AngleAxisd rotation_vector(M_PI/4, Vector3d(0,0,1)); // Z axie 45

    rotation_matrix = rotation_vector.toRotationMatrix();
    cout.precision(3);
    cout << "rotation vector to matrix is = \n" <<rotation_matrix << endl;

    Vector3d v(1,0,0);
    Vector3d v_rotated = rotation_vector*v;
    v_rotated = rotation_matrix*v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;

    Vector3d euler_angles = rotation_matrix  .eulerAngles(2,1,0); // ZYX shunxu
    cout << "Z Y X angle is " << euler_angles.transpose() << endl;

    Isometry3d T = Isometry3d::Identity();
    T.rotate(rotation_matrix);
    T.pretranslate(Vector3d(1,3,4));
    cout << "transform matrix = \n" << T.matrix() << endl;

    Vector3d v_transformed = T*v;
    cout << "v_transformed = " << v_transformed.transpose() << endl;

    // 四元数
    Quaterniond q = Quaterniond(rotation_vector);
    cout << "quaternion from rotation vector = " << q.coeffs().transpose() << endl;
    v_rotated = q*v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;
}

