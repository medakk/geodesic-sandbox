#include <iostream>
#include <string>
#include <chrono>
#include <limits>
#include <vector>
#include <robin_hood.h>

#include <vtkPLYReader.h>
#include <vtkPLYWriter.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkDijkstraGraphGeodesicPath.h>
#include <vtkSphereSource.h>
#include <vtkActor.h>
#include <vtkCellLocator.h>
#include <vtkProperty.h>
#include <vtkDoubleArray.h>
#include <random>
#include <igl/exact_geodesic.h>
#include <igl/heat_geodesics.h>
#include <igl/readPLY.h>


struct Vec3d {
  double x, y, z;
};

class GeodesicMethod {
public:
  virtual double distance(int i, int j) = 0;
  virtual size_t numVerts() const = 0;
  virtual Vec3d point(int i) const = 0;
};

struct IGLMesh {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
};

class VTKDijkstra: public GeodesicMethod {
  vtkSmartPointer<vtkPolyData> mesh;

public:
  VTKDijkstra(const std::string& plyFilePath)
  {
    auto reader = vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName(plyFilePath.c_str());
    reader->Update();
    mesh = reader->GetOutput();

    std::cerr << "VTKDijkstra: loaded mesh with " << numVerts() << " vertices" << std::endl;
  }

  virtual size_t numVerts() const override {
    return mesh->GetNumberOfPoints();
  }

  virtual Vec3d point(int i) const override {
    Vec3d out;
    mesh->GetPoint(i, (double *) &out);

    return out;
  }

  virtual double distance(int i, int j) override {
    auto dijkstra = vtkSmartPointer<vtkDijkstraGraphGeodesicPath>::New();
    dijkstra->SetInputData(mesh);
    dijkstra->SetStartVertex(i);
    dijkstra->SetEndVertex(j);
    dijkstra->Update();

    auto weights = vtkSmartPointer<vtkDoubleArray>::New();
    dijkstra->GetCumulativeWeights(weights);
    return weights->GetValue(j);
  }
};

class IGLExact: public GeodesicMethod {
  IGLMesh iglMesh;

public:
  IGLExact(const std::string& filepath)
  {
    igl::readPLY(filepath, iglMesh.V, iglMesh.F);

    std::cerr << "IGLExact: loaded mesh with " << numVerts() << " vertices" << std::endl;
  }

  virtual size_t numVerts() const override {
    return iglMesh.V.rows();
  }

  virtual Vec3d point(int i) const override {
    Vec3d out = {
      iglMesh.V(i, 0),
      iglMesh.V(i, 1),
      iglMesh.V(i, 2)
    };
    return out;
  }

  virtual double distance(int i, int j) override {
    Eigen::VectorXi VS, FS, VT, FT;
    VS.resize(1);
    VS << i;
    VT.resize(1);
    VT << j;

    Eigen::VectorXd d;
    igl::exact_geodesic(iglMesh.V, iglMesh.F, VS, FS, VT, FT, d);

    return d(0);
  }
};

class IGLHeat: public GeodesicMethod {
  IGLMesh iglMesh;
  igl::HeatGeodesicsData<double> heatData;

public:
  IGLHeat(const std::string& filepath)
  {
    igl::readPLY(filepath, iglMesh.V, iglMesh.F);
    igl::heat_geodesics_precompute(iglMesh.V,iglMesh.F,heatData);

    std::cerr << "IGLHeat: loaded mesh with " << numVerts() << " vertices" << std::endl;

    /*
    Eigen::MatrixXd D(numVerts(), numVerts());
    Eigen::VectorXi gamma;
    Eigen::VectorXd d;
    gamma.resize(1);
    for(int i=0; i<numVerts(); i++) {
      gamma(0) = i;
      igl::heat_geodesics_solve(heatData,gamma,d);
      D.row(i) = d;
    }

    std::ofstream f("distance_matrix/femur_10982.bin", std::ios::out | std::ios::binary);
    f.write((const char*) D.data(), D.size() * sizeof(Eigen::MatrixXd::Scalar));
    f.close();
    std::cout << "wrote distance transforms" << std::endl;
     */
  }

  virtual size_t numVerts() const override {
    return iglMesh.V.rows();
  }

  virtual Vec3d point(int i) const override {
    Vec3d out = {
      iglMesh.V(i, 0),
      iglMesh.V(i, 1),
      iglMesh.V(i, 2)
    };
    return out;
  }

  virtual double distance(int i, int j) override {

    Eigen::VectorXi gamma;
    Eigen::VectorXd D;
    gamma.resize(1); gamma << i;
    igl::heat_geodesics_solve(heatData,gamma,D);
    return D(j);
  }
};

class IGLCachedHeat: public GeodesicMethod {
  IGLMesh iglMesh;
  Eigen::SparseMatrix<double> S;

public:
  IGLCachedHeat(const std::string& filepath, double stopDistance)
  {
    igl::readPLY(filepath, iglMesh.V, iglMesh.F);
    igl::HeatGeodesicsData<double> heatData;
    igl::heat_geodesics_precompute(iglMesh.V,iglMesh.F,heatData); //TODO Checkout heat parameter

    Eigen::MatrixXd D;
    const auto N = numVerts();
    D.resize(N, N);

    std::vector<Eigen::Triplet<double>> triplets;

#pragma omp parallel for
    for(int i=0; i<N; i++) {
      Eigen::VectorXd d;
      Eigen::VectorXi gamma(1);
      gamma(0) = i;
      igl::heat_geodesics_solve(heatData, gamma, d);
      D.col(i) = d;
    }

    S = (D.array() > stopDistance).select(0.0, D).sparseView();
    // S.resize(N, N);
    // S.setFromTriplets(triplets.begin(), triplets.end());

    std::cerr << "IGLCachedHeat: loaded mesh with " << numVerts() << " vertices" << std::endl;
    std::cerr << "S: " << S.nonZeros() <<  " / " << (10982*10982) << std::endl;
    throw std::runtime_error("enough");

    /*
    std::ofstream out_file;
    out_file.open("distance_matrix/femur_D.txt");
    out_file.precision(std::numeric_limits<double>::max_digits10);
    out_file << D;
    out_file.close();
    */
  }

  virtual size_t numVerts() const override {
    return iglMesh.V.rows();
  }

  virtual Vec3d point(int i) const override {
    Vec3d out = {
            iglMesh.V(i, 0),
            iglMesh.V(i, 1),
            iglMesh.V(i, 2)
    };
    return out;
  }

  virtual double distance(int i, int j) override {
    return S.coeff(i, j);
  }
};

double analytic_geo_dist(double lon0, double lat0, double lon1, double lat1) {
  // https://en.wikipedia.org/wiki/Great-circle_distance
  const double del_lon = std::abs(lon0 - lon1);
  const double del_lat = std::abs(lat0 - lat1);

  // for debugging
  const double x0 = std::sin(lat0)*std::sin(lat1) + std::cos(lat0)*std::cos(lat1)*std::cos(del_lon);
  const double central_angle = std::acos(x0);
  const double r = 1.0; // radius

  return r * central_angle;
}

// assumes r=1.0
Vec3d geo_to_cart(double lon, double lat) {
  return {
    1.0 * std::cos(lat) * cos(lon),
    1.0 * std::cos(lat) * sin(lon),
    1.0 * std::sin(lat)
  };
}

// assumes r=1.0
Vec3d cart_to_geo(const Vec3d& p) {
  return {
    std::atan2(p.y, p.x),
    std::asin(p.z / 1.0),
    1.0
  };
}

// Find closest point
vtkIdType closest_point(const Vec3d &p, vtkSmartPointer<vtkCellLocator> cellLocator, vtkPolyData* mesh) {
  double closestPoint[3];//the coordinates of the closest point will be returned here
  double closestPointDist2; //the squared distance to the closest point will be returned here
  vtkIdType cellIdx, p1_cell_idx; //the cell id of the cell containing the closest point will be returned here
  int subId; //this is rarely used (in triangle strips only, I believe)
  cellLocator->FindClosestPoint((double *)&p, closestPoint, cellIdx, subId, closestPointDist2);
  return mesh->GetCell(cellIdx)->GetPointId(0);
}

vtkSmartPointer<vtkActor> viz_point(const Vec3d &p, const Vec3d &color) {
  auto points = vtkSmartPointer<vtkPoints>::New();
  auto vertices = vtkSmartPointer<vtkCellArray>::New();
  const auto p_id = points->InsertNextPoint((double *) & p);
  vertices->InsertNextCell(1);
  vertices->InsertCellPoint(p_id);

  auto poly = vtkSmartPointer<vtkPolyData>::New();
  poly->SetPoints(points);
  poly->SetVerts(vertices);

  auto mapper2 = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper2->SetInputData(poly);
  auto actor2 = vtkSmartPointer<vtkActor>::New();
  actor2->SetMapper(mapper2);
  actor2->GetProperty()->SetColor((double *) &color);
  actor2->GetProperty()->SetPointSize(20.0);
  return actor2;
}

IGLMesh into_igl_mesh(vtkPolyData* mesh) {
  const auto n_verts = mesh->GetNumberOfPoints();
  const auto n_faces = mesh->GetNumberOfPolys();

  IGLMesh igl_mesh;
  igl_mesh.V.resize(n_verts, 3);
  igl_mesh.F.resize(n_faces, 3);

  for(int i=0; i<n_verts; i++) {
    double p[3];
    mesh->GetPoint(i, p);
    igl_mesh.V(i, 0) = p[0];
    igl_mesh.V(i, 1) = p[1];
    igl_mesh.V(i, 2) = p[2];
  }

  mesh->GetPolys()->InitTraversal();
  vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();
  int row = 0;
  while(mesh->GetPolys()->GetNextCell(idList)) {
    igl_mesh.F(row, 0) = idList->GetId(0);
    igl_mesh.F(row, 1) = idList->GetId(1);
    igl_mesh.F(row, 2) = idList->GetId(2);

    row++;
  }

  return igl_mesh;
}

void eigen_decomposition() {
  std::cout << "DOING EIGEN COMPO\n";
  const std::string path = "/scratch/karthik/projects/geodesic-sandbox/python/distance_matrix/femur_10982.bin";
  std::ifstream in(path, std::ios::in | std::ios::binary);
  Eigen::MatrixXd D(10982, 10982);
  in.read(reinterpret_cast<char *>(D.data()), 10982 * 10982 * sizeof(Eigen::MatrixXd::Scalar));
  in.close();


  const double stopDistance = 25.0;
  Eigen::SparseMatrix<double> S = (D.array() > stopDistance).select(0.0, D).sparseView();


  std::cout << "eigen time\n";
  Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> solver(S);
  if(solver.info() != Eigen::Success) {
    throw std::runtime_error("eigen solver did not succeed");
  }

  std::cout << solver.eigenvalues() << std::endl;


  throw std::runtime_error("thats all folks");
}


int not_main(int argc, char *argv[]) {
  eigen_decomposition();

  if(argc != 3) {
    std::cerr << "missing arg(s)" << std::endl;
    return 1;
  }

  GeodesicMethod* geodesicFunc;
  const std::string whichFunc = argv[1];
  const std::string plyFilePath = argv[2];
  if(whichFunc == "vtk_dijkstra") {
    geodesicFunc = new VTKDijkstra(plyFilePath);
  } else if(whichFunc == "igl_exact") {
    geodesicFunc = new IGLExact(plyFilePath);
  } else if(whichFunc == "igl_heat") {
    geodesicFunc = new IGLHeat(plyFilePath);
  } else if(whichFunc == "igl_cached_heat") {
    geodesicFunc = new IGLCachedHeat(plyFilePath, 20.0);
  } else {
    std::cerr << "invalid arg" << std::endl;
    return 1;
  }

  // For random point generation
  std::uniform_int_distribution<int> points_dist(0, geodesicFunc->numVerts());
  std::default_random_engine re;

  std::ifstream query_file("/scratch/karthik/projects/ShapeWorks/Examples/Python/Output/femur_mesh/shape_models/dq_onlyfinal.txt");

  // Benchmark loop
  // gonna be biased, these rands
  for(int i=0; i<1000; i++) {
    // const int idx0 = points_dist(re);
    // const int idx1 = points_dist(re);
    int idx0, idx1, discard;
    query_file >> idx0 >> idx1 >> discard;

    const Vec3d p0 = geodesicFunc->point(idx0);
    const Vec3d p1 = geodesicFunc->point(idx1);

    Vec3d geo0 = cart_to_geo(p0);
    Vec3d geo1 = cart_to_geo(p1);
    const double lon0 = geo0.x;
    const double lat0 = geo0.y;
    const double lon1 = geo1.x;
    const double lat1 = geo1.y;
    const double analytic_soln = analytic_geo_dist(lon0, lat0, lon1, lat1);

    // Find closest point
    using namespace std::chrono;
    const auto startTime = high_resolution_clock::now();
    const double soln = geodesicFunc->distance(idx0, idx1);
    const auto endTime = high_resolution_clock::now();

    // Print time taken
    std::cout << duration_cast<nanoseconds>(endTime - startTime).count() << " " << soln << " ";

    // Print error
    const auto diff = std::abs(soln - analytic_soln);
    std::cout << diff << " ";

    std::cout << std::endl;
  }
  return 0;


  ////////////////////////////////////////////////////////
  // Visualization stuff
  /*
  vtkSmartPointer<vtkPolyDataMapper> mapper =
          vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputData(mesh);

  vtkSmartPointer<vtkActor> actor =
          vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
  vtkSmartPointer<vtkActor> actorWire =
          vtkSmartPointer<vtkActor>::New();
  actorWire->SetMapper(mapper);
  actorWire->GetProperty()->SetRepresentationToWireframe();
  actorWire->GetProperty()->SetColor(0.0, 0.0, 0.0);

  vtkSmartPointer<vtkRenderer> renderer =
          vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renderWindow =
          vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  renderWindow->SetSize(700, 700);
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
          vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);

  renderer->AddActor(actor);
  renderer->AddActor(actorWire);

  // stuff for debugging geodesic distance
  {
    const double lon = M_PI*3.0/4.0;
    const double lat = M_PI*1.0/4.0;
    const auto p = geo_to_cart(lon, lat);

    const auto idx_closest = closest_point(p, cellLocator, mesh);
    Vec3d pt_closest;
    mesh->GetPoint(idx_closest, (double *)&pt_closest);

    renderer->AddActor(viz_point(p, {1.0, 0.0, 0.0}));
    renderer->AddActor(viz_point(pt_closest, {0.0, 1.0, 0.0}));
  }

  renderWindow->Render();
  renderWindowInteractor->Start();
 */

	return 0;
}

/*
void viz_dijkstra()
{

  auto dijkstra = vtkSmartPointer<vtkDijkstraGraphGeodesicPath>::New();
  dijkstra->SetInputData(mesh);
  dijkstra->SetStartVertex(0);
  dijkstra->SetEndVertex(10);
  dijkstra->Update();
  const auto path = dijkstra->GetOutput();

  auto weights = vtkSmartPointer<vtkDoubleArray>::New();
  dijkstra->GetCumulativeWeights(weights);
  std::cout << "Geodesic distance _maybe_ " << weights->GetValue(10) << "\n";

  // Visualization stuff
  vtkSmartPointer<vtkPolyDataMapper> mapper =
          vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputData(mesh);

  vtkSmartPointer<vtkActor> actor =
          vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);

  // Create a mapper and actor
  vtkSmartPointer<vtkPolyDataMapper> pathMapper =
          vtkSmartPointer<vtkPolyDataMapper>::New();
  pathMapper->SetInputConnection(dijkstra->GetOutputPort());

  vtkSmartPointer<vtkActor> pathActor =
          vtkSmartPointer<vtkActor>::New();
  pathActor->SetMapper(pathMapper);
  pathActor->GetProperty()->SetColor(1,0,0); // Red
  pathActor->GetProperty()->SetLineWidth(7);

  vtkSmartPointer<vtkRenderer> renderer =
          vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renderWindow =
          vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  renderWindow->SetSize(700, 700);
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
          vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);

  renderer->AddActor(actor);
  renderer->AddActor(pathActor);

  renderWindow->Render();
  renderWindowInteractor->Start();
}
 */