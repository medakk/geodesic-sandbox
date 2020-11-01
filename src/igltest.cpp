#include <iostream>
#include <igl/readPLY.h>
#include <igl/hessian.h>
#include <igl/grad.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>

int main() {
  const auto mesh_filepath = "/scratch/karthik/projects/ShapeWorks/Examples/Python/TestFemurMesh/femur/meshes/m03_L_femur.ply";
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::readPLY(mesh_filepath, V, F);

  std::cout << "Mesh:\n";
  std::cout << "V: " << V.rows() << "x" << V.cols() << "\n";
  std::cout << "F: " << F.rows() << "x" << F.cols() << "\n";

  // compute per-vertex normals
  Eigen::MatrixXd N;
  igl::per_vertex_normals(V, F, N);

  // Lets keep only the X coordinate of the normal
  const Eigen::VectorXd U = N.col(0);

  // Find gradient operator
  Eigen::SparseMatrix<double> G;
  igl::grad(V, F, G);
  std::cout << "G: " << G.rows() << "x" << G.cols() << "\n";

  // find gradient of U
  const Eigen::MatrixXd GU = Eigen::Map<const Eigen::MatrixXd>((G*U).eval().data(),F.rows(),3);
  std::cout << "GU: " << GU.rows() << "x" << GU.cols() << "\n";

  // Compute gradient magnitude
  const Eigen::VectorXd GU_mag = GU.rowwise().norm();


  // Find gradient of U per vertex
  Eigen::MatrixXd GU_pervertex;
  igl::per_vertex_normals(V, F, GU, GU_pervertex);
  std::cout << "GU_pervertex: " << GU_pervertex.rows() << "x" << GU_pervertex.cols() << "\n";
  const Eigen::VectorXd GU_pervertex_mag = GU_pervertex.rowwise().norm();

  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  viewer.data().set_data(U);

  // Average edge length divided by average gradient (for scaling)
  const double max_size = igl::avg_edge_length(V,F) / GU_pervertex_mag.mean();
  // Draw a black segment in direction of gradient at face barycenters
  // Eigen::MatrixXd BC;
  // igl::barycenter(V,F,BC);
  const Eigen::RowVector3d black(0,0,0);
  viewer.data().add_edges(V,V+max_size*GU_pervertex, black);

  viewer.data().show_lines = false;
  viewer.data().line_width = 2.0;

  viewer.launch();
}