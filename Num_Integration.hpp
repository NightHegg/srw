
#include <cmath>
#include <vector>
#include <string>
#include "classes.hpp"

using namespace std;

void Create_Function(int dimTask,
					 VectorSchwarz &mesh,
					 double node,
					 int numElem,
					 MatrixSchwarz &D,
					 strainMatrix &S,
					 MatrixSchwarz &resMatrix)
{
	basfuncMatrix N(dimTask, mesh, numElem, node);
	MatrixSchwarz B;
	MatrixSchwarz BTD;
	MatrixSchwarz BT;
	B=S*N;
	B.Transpose(BT);
	BTD = BT * D;
	resMatrix = BTD * B;
	resMatrix = resMatrix * node;
}

void Numerical_Integration(int dimTask,
						   int numElem,
						   VectorSchwarz &mesh,
						   VectorSchwarz &elements,
						   MatrixSchwarz &D,
						   strainMatrix &S,
						   string typeIntegration,
						   MatrixSchwarz &ResMat)
{
	double h = mesh[numElem + 1] - mesh[numElem];
	std::vector<double> arrNodes;
	if (typeIntegration == "Riemann_Type")
	{
		arrNodes.push_back((mesh[numElem] + mesh[numElem + 1]) / 2.0);
		Create_Function(dimTask, mesh, arrNodes[0], numElem, D, S, ResMat);
		ResMat = ResMat * h;
	}
	else if (typeIntegration == "Trapezoidal_Type")
	{
		arrNodes.push_back(mesh[numElem]);
		arrNodes.push_back(mesh[numElem + 1]);

		MatrixSchwarz M1;
		MatrixSchwarz M2;

		Create_Function(dimTask, mesh, arrNodes[0], numElem, D, S, M1);
		Create_Function(dimTask, mesh, arrNodes[1], numElem, D, S, M2);

		ResMat = M1 + M2;
		ResMat = ResMat * (h / 2.0);
	}
	else if (typeIntegration == "Gauss_2_Type")
	{

		arrNodes.push_back((mesh[numElem] + mesh[numElem + 1]) / 2.0 - (mesh[numElem + 1] - mesh[numElem]) / (2 * sqrt(3.0)));
		arrNodes.push_back((mesh[numElem] + mesh[numElem + 1]) / 2.0 + (mesh[numElem + 1] - mesh[numElem]) / (2 * sqrt(3.0)));

		MatrixSchwarz M1;
		MatrixSchwarz M2;

		Create_Function(dimTask, mesh, arrNodes[0], numElem, D, S, M1);
		Create_Function(dimTask, mesh, arrNodes[1], numElem, D, S, M2);

		ResMat = M1 + M2;
		ResMat = ResMat * (h / 2.0);
	}
	else
	{
		printf("You entered the wrong type of numerical integration: %s\n", typeIntegration.c_str());
		exit(0);
	}
}