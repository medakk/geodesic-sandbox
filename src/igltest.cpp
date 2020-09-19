#include <iostream>
#include <igl/readPLY.h>
#include <igl/hessian.h>
#include <igl/grad.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>

int not_main() {
  const auto mesh_filepath = "/scratch/karthik/projects/ShapeWorks/Examples/Python/TestFemurMesh/femur/meshes/m03_L_femur.ply";
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::readPLY(mesh_filepath, V, F);

  std::cout << "V: " << V.rows() << "x" << V.cols() << "\n";
  std::cout << "F: " << F.rows() << "x" << F.cols() << "\n";

  Eigen::MatrixXd N_faces;
  igl::per_face_normals(V, F, N_faces);

  Eigen::SparseMatrix<double> G;
  igl::grad(V, F, G, false);
  std::cout << "G: " << G.rows() << "x" << G.cols() << "\n";

  /*
  const Eigen::MatrixXd x = G * V;
  std::cout<< "x: " << x.rows() << "x" <<x.cols() << "\n";

  const auto real_n0 = N_faces.row(1);
  std::cout << "Real n0: " << real_n0(0) << " " << real_n0(1) << " " << real_n0(2) << "\n";
  const auto x0 = x.row(0) + x.row(1) + x.row(2);
  std::cout << "x0" << x0(0) << " " << x0(1) << " " << x0(2) << "\n";
  */

  Eigen::MatrixXd N_verts;
  igl::per_vertex_normals(V, F, N_verts);
  const auto GN = G * N_verts;
  std::cout << "GN: " << GN.rows() << "x" << GN.cols() << std::endl;

  Eigen::SparseMatrix<double> H;
  igl::hessian(V, F, H);

  std::cout << "H: " << H.rows() << "x" << H.cols() << "\n";
}