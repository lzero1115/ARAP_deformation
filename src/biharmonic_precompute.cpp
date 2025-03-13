//
// Created by Aolilex on 2025/03/12.
//
#include "biharmonic_precompute.h"
#include <igl/min_quad_with_fixed.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>

void biharmonic_precompute(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  const Eigen::VectorXi & b, // fixed node indices
  igl::min_quad_with_fixed_data<double> & data) {

        Eigen::SparseMatrix<double> L, M, M_inv, I, Q, Aeq;
        igl::cotmatrix(V,F,L); // semi-negative definite
        igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_DEFAULT,M);
        int nv = V.rows();
        I.resize(nv,nv);
        I.setIdentity();
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(M);
        M_inv = solver.solve(I);
        Q.resize(nv, nv);
        Q = L.transpose() * M_inv * L; // L^T M^-T M M^-1 L
        // min 0.5 z^T * Q *z + z^T*B
        // s.t. zb = zbc, Aeq*z = Beq
        igl::min_quad_with_fixed_precompute(Q, b, Aeq, true, data);


}