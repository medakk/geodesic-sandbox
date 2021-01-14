#include <iostream>
#include <igl/readPLY.h>
#include <igl/grad.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/heat_geodesics.h>
#include <Eigen/Dense>
#include <cstdlib>
#include <random>
#include <unordered_map>


std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<> rand01(0, 1);

Eigen::MatrixXd V;
Eigen::MatrixXi F;
igl::HeatGeodesicsData<double> heat_data;

using Vec3 = Eigen::Vector3d;
void sample_point(Vec3& out, Vec3& bary, int& f) {
  f = rand() % F.rows();
  const auto v0 = F(f, 0);
  const auto v1 = F(f, 1);
  const auto v2 = F(f, 2);

  const double b0 = rand01(e2);
  const double b1 = rand01(e2) * (1.0 - b0);
  const double b2 = 1.0 - b1 - b0;

  out = b0*V.row(v0) + b1*V.row(v1) + b2*V.row(v2);
  bary = {b0, b1, b2};
}


double geodesic_distance(int v1, int v2) {
  static std::unordered_map<int, Eigen::VectorXd> cache_v;
  if(v1 > v2) {
    std::swap(v1, v2);
  }

  const auto entry = cache_v.find(v1);
  if(entry != cache_v.end()) {
    return entry->second(v2);
  }

  Eigen::VectorXi gamma;
  Eigen::VectorXd D;
  gamma.resize(1); gamma << v1;
  igl::heat_geodesics_solve(heat_data, gamma, D);

  const double d = D(v2);
  cache_v[v1] = std::move(D);

  return d;
}

double geodesic_distance_from_face(int f, int v) {
  static std::unordered_map<int, Eigen::VectorXd> cache_f;
  const auto entry = cache_f.find(f);
  if(entry != cache_f.end()) {
    return entry->second(v);
  }

  Eigen::VectorXi gamma;
  Eigen::VectorXd D;
  gamma.resize(1); gamma << F(f, 0), F(f, 1), F(f, 2);
  igl::heat_geodesics_solve(heat_data, gamma, D);

  const double d = D(v);
  cache_f[f] = std::move(D);

  return d;
}

// consider 9 possible paths based on triangles and chose the shortest
double geodesic_distance_bo9(
        const Vec3& pt_a, const Vec3& bary_a, int f_a,
        const Vec3& pt_b, const Vec3& bary_b, int f_b
        ) {
  if(f_a == f_b) {
    return (pt_a - pt_b).norm();
  }

  double best_dist = 1e9;
  for(int i=0; i<3; i++) {
    // geodesic(==euclidean) distance between point a and face i
    const double d_ai = (pt_a - V.row(F(f_a, i)).transpose()).norm();

    for(int j=0; j<3; j++) {
      // geodesic distance between the two vertices on the mesh
      const double g_ij = geodesic_distance(F(f_a,i), F(f_b,j));

      // geodesic(==euclidean) distance between point b and face j
      const double d_bi = (pt_b - V.row(F(f_b, j)).transpose()).norm();

      // total geodesic distance
      const double g = d_ai + g_ij + d_bi;

      // std::cout << "(" << i << "," << j <<") " << d_ai << " + " << g_ij << " + " << d_bi << " == " << g << "\n";

      if(g < best_dist) {
        best_dist = g;
      }
    }
  }

  return best_dist;
}

// barycentric, but my own interpretation?
double geodesic_distance_barykk(
  const Vec3& pt_a, const Vec3& bary_a, int f_a,
  const Vec3& pt_b, const Vec3& bary_b, int f_b
  ) {
  if(f_a == f_b) {
    return (pt_a - pt_b).norm();
  }

  const Vec3 distances = {
          geodesic_distance_from_face(f_b, F(f_a, 0)),
          geodesic_distance_from_face(f_b, F(f_a, 1)),
          geodesic_distance_from_face(f_b, F(f_a, 2)),
  };
  return distances.dot(bary_a);
}

int main() {
  srand(time(nullptr));

  const auto mesh_filepath = "/scratch/karthik/projects/ShapeWorks/Examples/Python/Output/plane/ply/plane_highres.ply";
  igl::readPLY(mesh_filepath, V, F);

  std::cout << "Mesh:\n";
  std::cout << "V: " << V.rows() << "x" << V.cols() << "\n";
  std::cout << "F: " << F.rows() << "x" << F.cols() << "\n";
  std::cout << "Bounds: [" << V.minCoeff() << "] -> [" << V.maxCoeff() << "]\n\n";

  igl::heat_geodesics_precompute(V, F, heat_data);

  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  // viewer.data().set_data(D);

  Vec3 pt_a, bary_a; int f_a;
  Vec3 pt_b, bary_b; int f_b;

  for(int i=0; i<10000; i++) {
    sample_point(pt_a, bary_a, f_a);
    sample_point(pt_b, bary_b, f_b);
    // const double geo = geodesic_distance_bo9(pt_a, bary_a, f_a, pt_b, bary_b, f_b);
    const double geo = geodesic_distance_barykk(pt_a, bary_a, f_a, pt_b, bary_b, f_b);
    const double euc = (pt_a - pt_b).norm();
    std::cout << "distance: " << geo << " | " << euc << " (diff: " << geo - euc << ")\n";
  }

  Eigen::MatrixXd T1(2, 3);
  T1.row(0) = pt_a;
  T1.row(1) = pt_b;

  Eigen::MatrixXd T2(1, 3); T2 << 1.0, 0.0, 0.0;
  viewer.data().set_points(T1, T2);

  // viewer.data().show_lines = false;
  // viewer.data().line_width = 2.0;
  viewer.launch();

  return 0;
}
