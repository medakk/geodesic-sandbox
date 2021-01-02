#include <iostream>
#include <igl/readPLY.h>
#include <igl/hessian.h>
#include <igl/grad.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/doublearea.h>
#include <igl/heat_geodesics.h>

void into_per_vertex(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& FN, Eigen::MatrixXd& N) {
  using namespace std;
  // Resize for output
  N.setZero(V.rows(),3);

  Eigen::MatrixXd W(F.rows(),3);
  {
    Eigen::MatrixXd A(F.rows(), 1);
    igl::doublearea(V,F,A);
    W = A.replicate(1,3);
  }

  // loop over faces
  for(int i = 0;i<F.rows();i++)
  {
    // throw normal at each corner
    for(int j = 0; j < 3;j++)
    {
      N.row(F(i,j)) += W(i,j) * FN.row(i);
    }
  }

  N.rowwise().normalize();
}

int NORMAL_main() {
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
#if 0
  const Eigen::VectorXd U = N.col(0);
  {
    std::ofstream f("/scratch/karthik/projects/geodesic-sandbox/python/random/libigl_normalx.bin");
    f.write(reinterpret_cast<const char *>(U.data()), U.size() * sizeof(double));
  }
#endif

  // Find gradient operator
  Eigen::SparseMatrix<double> G;
  igl::grad(V, F, G);
  std::cout << "G: " << G.rows() << "x" << G.cols() << "\n";

  // find gradient of U
  const Eigen::MatrixXd GU_all = Eigen::Map<const Eigen::MatrixXd>((G*N).eval().data(),F.rows(),9);
  std::cout << "GU_all: " << GU_all.rows() << "x" << GU_all.cols() << "\n";

  const Eigen::MatrixXd GU = GU_all.block(0, 0, GU_all.rows(), 3);
  // const Eigen::MatrixXd GU_truth = Eigen::Map<const Eigen::MatrixXd>((G*N.col(0)).eval().data(),F.rows(),3);

  // Compute gradient magnitude
  const Eigen::VectorXd GU_mag = GU.rowwise().norm();


  // Find gradient of U per vertex

  Eigen::MatrixXd GU_pervertex;
  // igl::per_vertex_normals(V, F, GU, GU_pervertex);
  into_per_vertex(V, F, GU, GU_pervertex);
  std::cout << "GU_pervertex: " << GU_pervertex.rows() << "x" << GU_pervertex.cols() << "\n";

  const Eigen::VectorXd GU_pervertex_mag = GU_pervertex.rowwise().norm();

  /*
  std::ofstream f("/scratch/karthik/projects/geodesic-sandbox/python/random/libigl_gradx.bin", std::ios::binary | std::ios::out);
  f.write(reinterpret_cast<const char *>(GU_pervertex.data()), GU_pervertex.size() * sizeof(double));
  f.close();
   */

  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  const Eigen::VectorXd Nx = N.col(0);
  viewer.data().set_data(Nx);

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

int main() {
  const auto mesh_filepath = "/scratch/karthik/projects/ShapeWorks/Examples/Python/TestFemurMesh/femur/meshes/m03_L_femur.ply";
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::readPLY(mesh_filepath, V, F);

  std::cout << "Mesh:\n";
  std::cout << "V: " << V.rows() << "x" << V.cols() << "\n";
  std::cout << "F: " << F.rows() << "x" << F.cols() << "\n";

  // compute per-vertex geodesics from vertex 0
  igl::HeatGeodesicsData<double> heat_data;
  igl::heat_geodesics_precompute(V, F, heat_data);
  Eigen::VectorXi gamma; gamma.resize(1); gamma << (V.rows()/2);
  Eigen::VectorXd N;
  igl::heat_geodesics_solve(heat_data,gamma,N);

  // Find gradient operator
  Eigen::SparseMatrix<double> G;
  igl::grad(V, F, G);
  std::cout << "G: " << G.rows() << "x" << G.cols() << "\n";

  // find gradient of U
  const Eigen::MatrixXd GU_all = Eigen::Map<const Eigen::MatrixXd>((G*N).eval().data(),F.rows(),3);
  std::cout << "GU_all: " << GU_all.rows() << "x" << GU_all.cols() << "\n";

  const Eigen::MatrixXd GU = GU_all.block(0, 0, GU_all.rows(), 3);
  // const Eigen::MatrixXd GU_truth = Eigen::Map<const Eigen::MatrixXd>((G*N.col(0)).eval().data(),F.rows(),3);

  // Compute gradient magnitude
  const Eigen::VectorXd GU_mag = GU.rowwise().norm();


  // Find gradient of U per vertex

  Eigen::MatrixXd GU_pervertex;
  // igl::per_vertex_normals(V, F, GU, GU_pervertex);
  into_per_vertex(V, F, GU, GU_pervertex);
  std::cout << "GU_pervertex: " << GU_pervertex.rows() << "x" << GU_pervertex.cols() << "\n";

  const Eigen::VectorXd GU_pervertex_mag = GU_pervertex.rowwise().norm();

  /*
  std::ofstream f("/scratch/karthik/projects/geodesic-sandbox/python/random/libigl_gradx.bin", std::ios::binary | std::ios::out);
  f.write(reinterpret_cast<const char *>(GU_pervertex.data()), GU_pervertex.size() * sizeof(double));
  f.close();
   */

  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  const Eigen::VectorXd Nx = N.col(0);
  viewer.data().set_data(Nx);

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
