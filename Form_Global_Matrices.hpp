#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include "Num_Integration.hpp"
#include "Methods.hpp"
#include "classes.hpp"
#include "record_data.hpp"
#include "AddFuncs_Ensembling.hpp"
#include "AddFuncs_BoundaryConds.hpp"

using namespace std;

void SolveLinearSystem(int dimTask, MatrixSchwarz &K, VectorSchwarz &F, VectorSchwarz &y)
{
	switch (dimTask)
	{
	case 1:
		TridiogonalMatrixAlgorithmRight(K, F, y);
		break;
	case 2:
		GaussianElimination(K, F, y);
		break;
	}
}

void CalcDisplacements(int 				dimTask,
					   string 			*Route,
					   VectorSchwarz 	&y,
					   VectorSchwarz 	&mesh,
					   VectorSchwarz 	&elements,
					   strainMatrix 	&S,
					   MatrixSchwarz 	&D,
					   int 				uk,
					   int 				amntSubdomains,
					   double 			stopCriteria,
					   int 				amntNodes,
					   int 				amntElements)
{
	int Counter{0};
	double coefOverlap{0};

	MatrixSchwarz K;
	VectorSchwarz F;

	std::stringstream ss;
	ss << stopCriteria;
	ss.precision(7);
	std::string sStopCriteria = ss.str();

	VectorSchwarz yPrevious(mesh.GetSize());

	VectorSchwarz meshSubd;
	VectorSchwarz ySubd;
	VectorSchwarz yPreviousSubd;

	double tmp{0};
	int size = mesh.GetSize();
	vector<double> arrBound;
	ifstream scan("files/" + to_string(dimTask) + "D/boundaryConditions.dat");
	while (!scan.eof())
	{
		scan >> tmp;
		arrBound.push_back(tmp);
	}
	if (amntSubdomains < 2)
	{
		*Route += "Non_Schwarz/";
		K.Construct(amntNodes * dimTask, amntNodes * dimTask);
		F.Construct(amntNodes * dimTask);

		Ensembling(dimTask, K, F, D, S, mesh, elements, amntNodes, amntElements);

		Form_Boundary_Conditions(dimTask, arrBound, y, mesh, elements, amntElements, amntNodes, K, F);
		//K.SetName("K");
		//F.SetName("F");
		//K.Record("",1);
		//F.Record("",1);

		SolveLinearSystem(dimTask, K, F, y);
	}
	else
	{
		y.Fill(-1e-6);
		*Route += "Schwarz/SC_" + sStopCriteria + "/";
		mesh.Decomposition(amntSubdomains, &coefOverlap);
		y.Equal_SchwarzNodes(mesh);
		yPrevious.Equal_SchwarzNodes(mesh);
		do
		{
			yPrevious = y;
			for (int i = 0; i < amntSubdomains; i++)
			{
				meshSubd = mesh.CreateAllocatedArray(i);
				ySubd = y.CreateAllocatedArray(i);
				yPreviousSubd = yPrevious.CreateAllocatedArray(i);

				amntNodes = meshSubd.GetSize();
				amntElements = amntNodes - 1;

				K.Construct(meshSubd.GetSize(), meshSubd.GetSize());
				F.Construct(meshSubd.GetSize());

				Ensembling(dimTask, K, F, D, S, meshSubd, elements, amntNodes, amntElements);

				Form_Boundary_Conditions_Schwarz(dimTask, arrBound, y, ySubd, yPreviousSubd, meshSubd, K, F);

				SolveLinearSystem(dimTask, K, F, y);

				y.ReturnAllocatedArrayResults(ySubd, i);

				K.~MatrixSchwarz();
				F.~VectorSchwarz();
			}
			Counter++;
		} while (y.ConvergenceL2(yPrevious, mesh) > stopCriteria);
		printf("Amount of iterations: %d\n\n", Counter);
	}
	y.SetName("y");

	FormRouteSchwarz(Route, amntNodes, amntSubdomains);

	y.Record(*Route, uk);

	//Record_AddData(mesh.GetSize(), Route, amntSubdomains, Counter, stopCriteria, coefOverlap);
}