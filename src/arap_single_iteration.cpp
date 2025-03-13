//
// Created by Aolilex on 2025/03/12.
//
//
// Created by Aolilex on 2025/03/12.
//
#include "arap_single_iteration.h"
#include <igl/polar_svd3x3.h>
#include <igl/min_quad_with_fixed.h>

void arap_single_iteration(
  const igl::min_quad_with_fixed_data<double> & data,
  const Eigen::SparseMatrix<double> & K,
  const Eigen::MatrixXd & bc,
  Eigen::MatrixXd & U)
{
    Eigen::MatrixXd R_T;
    Eigen::MatrixXd C = K.transpose() * U; // 3nx3
    R_T.resizeLike(C.transpose());
    for(int i=0;i<data.n;i++)
    {
        Eigen::Matrix3d Ck = C.block<3,3>(i*3,0);
        Eigen::Matrix3d Rk;
        igl::polar_svd3x3(Ck,Rk);
        R_T.block<3,3>(0,3*i) = Rk.transpose();
    }

    // Explicitly compute the product to avoid template instantiation issues
    Eigen::MatrixXd B = K * R_T.transpose();

    Eigen::MatrixXd Beq;
    igl::min_quad_with_fixed_solve(data, B, bc, Beq, U);
}
