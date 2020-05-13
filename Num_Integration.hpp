
#include <cmath>
#include <vector>
#include <string>
#include "Classes_Schwarz.hpp"
#include "classes/Basis_Functions.hpp"

using namespace std;

void Create_Function(int dimTask,
					 VectorSchwarz &a,
					 double node,
					 int numElem,
					 MatrixSchwarz &D,
					 strainMatrix &S,
					 MatrixSchwarz &resMatrix)
{
	basfuncMatrix N(dimTask, a, numElem, node);
	MatrixSchwarz B;
	MatrixSchwarz BTD;
	MatrixSchwarz BT;
	B = S * N;
	B.ConstructFullB(dimTask, a, node, numElem);
	B.Transpose(BT);
	BTD = BT * D;
	resMatrix = BTD * B;
	resMatrix = resMatrix * node;
}

void Numerical_Integration(int dimTask,
						   int numElem,
						   VectorSchwarz &a,
						   MatrixSchwarz &D,
						   strainMatrix &S,
						   string typeIntegration,
						   MatrixSchwarz &ResMat)
{
	double h = a[numElem + 1] - a[numElem];
	std::vector<double> arrNodes;
	if (typeIntegration == "Riemann_Type")
	{
		arrNodes.push_back((a[numElem] + a[numElem + 1]) / 2.0);
		Create_Function(dimTask, a, arrNodes[0], numElem, D, S, ResMat);
		ResMat = ResMat * h;
	}
	else if (typeIntegration == "Trapezoidal_Type")
	{
		arrNodes.push_back(a[numElem]);
		arrNodes.push_back(a[numElem + 1]);

		MatrixSchwarz M1;
		MatrixSchwarz M2;

		Create_Function(dimTask, a, arrNodes[0], numElem, D, S, ResMat);
		Create_Function(dimTask, a, arrNodes[1], numElem, D, S, ResMat);

		ResMat = M1 + M2;
		ResMat = ResMat * (h / 2.0);
	}
	else if (typeIntegration == "Gauss_2_Type")
	{

		arrNodes.push_back((a[numElem] + a[numElem + 1]) / 2.0 - (a[numElem + 1] - a[numElem]) / (2 * sqrt(3.0)));
		arrNodes.push_back((a[numElem] + a[numElem + 1]) / 2.0 + (a[numElem + 1] - a[numElem]) / (2 * sqrt(3.0)));

		MatrixSchwarz M1;
		MatrixSchwarz M2;

		Create_Function(dimTask, a, arrNodes[0], numElem, D, S, ResMat);
		Create_Function(dimTask, a, arrNodes[1], numElem, D, S, ResMat);

		ResMat = M1 + M2;
		ResMat = ResMat * (h / 2.0);
	}
	else
	{
		printf("You entered the wrong type of numerical integration: %s\n", typeIntegration.c_str());
		exit(0);
	}
}