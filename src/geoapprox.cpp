#include <iostream>
#include <igl/readPLY.h>
#include <igl/grad.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/heat_geodesics.h>
#include <igl/triangle_triangle_adjacency.h>
#include <Eigen/Dense>
#include <cstdlib>
#include <random>
#include <unordered_map>
#include <chrono>


std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<> rand01(0, 1);

using Vec3 = Eigen::Vector3d;
Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXi TT;
Eigen::SparseMatrix<double> G;
igl::HeatGeodesicsData<double> heat_data;
std::unordered_map<int, std::set<int>> neighbours;

void ComputeNeighbours() {
  for(int f = 0; f< F.rows(); f++) {
    const auto f0 = F(f, 0);
    const auto f1 = F(f, 1);
    const auto f2 = F(f, 2);

    neighbours[f0].insert(f1);
    neighbours[f0].insert(f2);
    neighbours[f1].insert(f0);
    neighbours[f1].insert(f2);
    neighbours[f2].insert(f0);
    neighbours[f2].insert(f1);
  }
}

bool is_in_ring(int f_a, int f_b, int ring) {
  std::vector<int> q;
  std::set<int> visited;

  q.push_back(f_b);
  for(int i=0; i<ring; i++) {
    if(q.size() == 0) {
      break;
    }

    std::vector<int> new_q;
    for(auto &f : q) {
      if(visited.find(f) != visited.end()) {
        continue;
      }

      visited.insert(f);
      int n0 = TT(f, 0);
      int n1 = TT(f, 1);
      int n2 = TT(f, 2);
      if(n0 == f_a || n1 == f_a || n2 == f_a) {
        return true;
      }
      new_q.push_back(n0);
      new_q.push_back(n1);
      new_q.push_back(n2);
    }
    q = std::move(new_q);
  }
  return false;
}


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

// doesn't consider TWO POINTS ON SAME TRIANGLE
void sample_point_close_to(int f_orig, Vec3& out, Vec3& bary, int& f) {
  const int hops = rand() % 5;
  f = f_orig;
  for(int i=0; i<hops || f==f_orig; i++) {
    int new_f = -1;
    while(new_f == -1) {
      new_f = TT(f, rand()%3);
    }
    f = new_f;
  }

  const auto v0 = F(f, 0);
  const auto v1 = F(f, 1);
  const auto v2 = F(f, 2);

  const double b0 = rand01(e2);
  const double b1 = rand01(e2) * (1.0 - b0);
  const double b2 = 1.0 - b1 - b0;

  out = b0*V.row(v0) + b1*V.row(v1) + b2*V.row(v2);
  bary = {b0, b1, b2};
}

const Eigen::VectorXd& all_geodesics_from_point(int v1) {
  static std::unordered_map<int, Eigen::VectorXd> cache_v;

  const auto entry = cache_v.find(v1);
  if(entry != cache_v.end()) {
    return entry->second;
  }

  Eigen::VectorXi gamma;
  Eigen::VectorXd D;
  gamma.resize(1); gamma << v1;
  igl::heat_geodesics_solve(heat_data, gamma, D);
  cache_v[v1] = std::move(D);

  return cache_v[v1];
}

const Eigen::VectorXd all_euclidean_from_point(const Vec3& src) {
  Eigen::VectorXd D;

  D = (V.array().rowwise() - src.transpose().array()).matrix().rowwise().norm();

  return D;
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
        const Vec3& pt_b, const Vec3& bary_b, int f_b,
        Vec3* grad_out=nullptr
        ) {
  if(f_a == f_b) {
    if(grad_out != nullptr) {
      *grad_out = pt_a - pt_b;
    }
    return (pt_a - pt_b).norm();
  }

  int best_i, best_j;
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
        best_i = i;
        best_j = j;
      }
    }
  }

  if(grad_out != nullptr) {
    // const auto& D = all_geodesics_from_point(F(f_b, best_j));
    const auto D = all_euclidean_from_point(pt_b);
    const Eigen::MatrixXd GD_all = Eigen::Map<const Eigen::MatrixXd>((G*D).eval().data(), F.rows(), 3);
    const Eigen::MatrixXd GD = GD_all.row(f_a);

    (*grad_out)[0] = GD(0, 0);
    (*grad_out)[1] = GD(0, 1);
    (*grad_out)[2] = GD(0, 2);
  }

  return best_dist;
}

double geodesic_distance_barydatar(
        const Vec3& pt_a, const Vec3& bary_a, int f_a,
        const Vec3& pt_b, const Vec3& bary_b, int f_b,
        Vec3 *grad_out
) {
  // ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3346950/
  if(f_a == f_b) {
    if(grad_out != nullptr) {
      *grad_out = pt_a - pt_b;
    }
    return (pt_a - pt_b).norm();
  }

  double res = 0.0;

  for(int i=0; i<3; i++) {
    const double g_i0 = geodesic_distance(F(f_a,i), F(f_b,0));
    const double g_i1 = geodesic_distance(F(f_a,i), F(f_b,1));
    const double g_i2 = geodesic_distance(F(f_a,i), F(f_b,2));
    const double g_iy = bary_b[0]*g_i0 + bary_b[1]*g_i1 + bary_b[2]*g_i2;
    res += bary_a[i] * g_iy;
  }

  if(grad_out != nullptr) {
    if(is_in_ring(f_a, f_b, 1)) {
      *grad_out = pt_a - pt_b;
    } else {
      *grad_out = Vec3(0.0, 0.0, 0.0);
      const auto D = all_euclidean_from_point(pt_b);
      const Eigen::MatrixXd GD_all = Eigen::Map<const Eigen::MatrixXd>((G*D).eval().data(), F.rows(), 3);

      Eigen::MatrixXd GD_all_pervertex;
      igl::per_vertex_normals(V, F, GD_all, GD_all_pervertex);

      for(int i=0; i<3; i++) {
        const Vec3 g = GD_all_pervertex.row(F(f_a, i));
        *grad_out += bary_a[i] * g;
      }
    }
  }

  return res;
}

double geodesic_distance_testmeshfimgrad(
        const Vec3& pt_a, const Vec3& bary_a, int f_a,
        const Vec3& pt_b, const Vec3& bary_b, int f_b,
        Vec3 *grad_out
) {
  if(grad_out != nullptr) {
    *grad_out = {0.0, 0.0, 0.0};

    const auto D = all_euclidean_from_point(pt_b);
    const auto grad_v = [&](int v) {
      Vec3 df = {0.0, 0.0, 0.0};
      for(int o : neighbours[v]) {
        const Vec3 pt_diff = V.row(o) - V.row(v);
        const double value_diff = D[o] - D[v];

        df[0] += value_diff / (pt_diff[0] + 0.0001);
        df[1] += value_diff / (pt_diff[1] + 0.0001);
        df[2] += value_diff / (pt_diff[2] + 0.0001);
      }
      df /= (double) neighbours[v].size();
      return df;
    };

    for(int i=0; i<3; i++) {
      *grad_out += bary_a[i] * grad_v(F(f_a, i));
    }

  }

  return (pt_a - pt_b).norm();
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

  std::cerr << "Mesh:\n";
  std::cerr << "V: " << V.rows() << "x" << V.cols() << "\n";
  std::cerr << "F: " << F.rows() << "x" << F.cols() << "\n";
  std::cerr << "Bounds: [" << V.minCoeff() << "] -> [" << V.maxCoeff() << "]\n\n";

  igl::grad(V, F, G);
  igl::triangle_triangle_adjacency(F, TT);
  ComputeNeighbours();

  igl::heat_geodesics_precompute(V, F, heat_data);

  Vec3 pt_a, bary_a; int f_a;
  Vec3 pt_b, bary_b; int f_b;

  // compare geodesic distance
  for(int i=0; i<10000; i++) {
    sample_point(pt_a, bary_a, f_a);

    if(i < 5000) {
      sample_point_close_to(f_a, pt_b, bary_b, f_b);
    } else {
      sample_point(pt_b, bary_b, f_b);
    }

    Vec3 geo_grad;
    // const double geo = geodesic_distance_barykk(pt_a, bary_a, f_a, pt_b, bary_b, f_b);
    // const double geo = geodesic_distance_testmeshfimgrad(pt_a, bary_a, f_a, pt_b, bary_b, f_b, &geo_grad);
    // const double geo = geodesic_distance_bo9(pt_a, bary_a, f_a, pt_b, bary_b, f_b, &geo_grad);
    const double geo = geodesic_distance_barydatar(pt_a, bary_a, f_a, pt_b, bary_b, f_b, &geo_grad);

    /*
    using namespace std::chrono;
    const auto start = high_resolution_clock::now();
    const auto end = high_resolution_clock::now();
    const auto elapsed = duration_cast<nanoseconds>(end - start).count();
    std::cout << "time: " << elapsed <<"\n";
    continue;
    */

    const double euc = (pt_a - pt_b).norm();
    const Vec3 euc_grad = pt_a - pt_b;

    const double geo_angle = std::atan2(geo_grad.y(), geo_grad.x());
    const double euc_angle = std::atan2(euc_grad.y(), euc_grad.x());
    const double geo_mag = geo_grad.norm();
    const double euc_mag = euc_grad.norm();

    std::cout << "distance: " << geo <<       " | " << euc <<       " (diff: " << geo - euc <<             ") ";
    std::cout << "| angle:  " << geo_angle << " | " << euc_angle << " (diff: " << geo_angle - euc_angle << ") ";
    std::cout << "|   mag:  " << geo_mag   << " | " << euc_mag   << " (diff: " << geo_mag   - euc_mag   << ") ";
    std::cout << "\n";
  }

  return 0;

  Eigen::MatrixXd T1(2, 3);
  T1.row(0) = pt_a;
  T1.row(1) = pt_b;

  Eigen::MatrixXd T2(1, 3); T2 << 1.0, 0.0, 0.0;
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  viewer.data().set_points(T1, T2);
  // viewer.data().show_lines = false;
  // viewer.data().line_width = 2.0;
  viewer.launch();

  return 0;
}
