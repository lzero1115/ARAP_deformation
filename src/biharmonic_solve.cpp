//
// Created by Aolilex on 2025/03/12.
//
#include "biharmonic_solve.h"
#include <igl/min_quad_with_fixed.h>



void biharmonic_solve(
  const igl::min_quad_with_fixed_data<double> & data,
  const Eigen::MatrixXd & bc, // fixed node value
  Eigen::MatrixXd & D){

      Eigen::MatrixXd B = Eigen::MatrixXd::Zero(data.n, 1); // z^T*B
      Eigen::MatrixXd Beq; // Aeq*z = Beq
      // solve displacement field
      igl::min_quad_with_fixed_solve(data, B, bc, Beq,D);

 }