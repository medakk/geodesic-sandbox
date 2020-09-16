#include <iostream>
#include <string>

#include <vtkPLYReader.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkDijkstraGraphGeodesicPath.h>
#include <vtkSphereSource.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkDoubleArray.h>

int not_main() {
  // return vtk_stuff();
}

int vtk_stuff() {
  const auto mesh_filepath = "/scratch/karthik/projects/ShapeWorks/Examples/Python/TestFemurMesh/femur/meshes/m03_L_femur.ply";
  auto reader = vtkSmartPointer<vtkPLYReader>::New();
  reader->SetFileName(mesh_filepath);
  reader->Update();
  const auto mesh = reader->GetOutput();

  std::cout << "Verts:  " << mesh->GetNumberOfVerts() << "\n";
  std::cout << "Points: " << mesh->GetNumberOfPoints() << "\n";
  std::cout << "Lines:  " << mesh->GetNumberOfLines() << "\n";
  std::cout << "Polys:  " << mesh->GetNumberOfPolys() << "\n";

  /*
  mesh->GetPolys()->InitTraversal();
  auto idList = vtkSmartPointer<vtkIdList>::New();
  while(mesh->GetPolys()->GetNextCell(idList))
  {
    std::cout << "Poly has " << idList->GetNumberOfIds() << " points." << std::endl;
    for(vtkIdType i = 0; i < idList->GetNumberOfIds(); i++)
    {
      double v[3];
      mesh->GetPoint(idList->GetId(i), v);
      std::cout << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
  }
  */

  auto sphereSource = vtkSmartPointer<vtkSphereSource>::New();
  sphereSource->Update();
  const auto sphereMesh = sphereSource->GetOutput();
  std::cout << "Sphere: \n";
  std::cout << "Verts:  " << sphereMesh->GetNumberOfVerts() << "\n";
  std::cout << "Points: " << sphereMesh->GetNumberOfPoints() << "\n";
  std::cout << "Lines:  " << sphereMesh->GetNumberOfLines() << "\n";
  std::cout << "Polys:  " << sphereMesh->GetNumberOfPolys() << "\n";

  auto dijkstra = vtkSmartPointer<vtkDijkstraGraphGeodesicPath>::New();
  dijkstra->SetInputData(mesh);
  dijkstra->SetStartVertex(0);
  dijkstra->SetEndVertex(8000);
  dijkstra->Update();
  const auto path = dijkstra->GetOutput();

  auto weights = vtkSmartPointer<vtkDoubleArray>::New();
  dijkstra->GetCumulativeWeights(weights);
  std::cout << "Geodesic distance _maybe_ " << weights->GetValue(8000) << "\n";

  // return 0;

  ////////////////////////////////////////////////////////
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

	return 0;
}
