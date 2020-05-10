
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
					 MatrixSchwarz &resMatrix)
{
	Basis_Functions N(dimTask, a, numElem, node);
	MatrixStrain S
	MatrixSchwarz B;
	MatrixSchwarz BTD;
	MatrixSchwarz BT;
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
						   string Type_Integration,
						   MatrixSchwarz &ResMat)
{
	double h = a[numElem + 1] - a[numElem];
	std::vector<double> arrNodes;
	if (Type_Integration == "Riemann_Type")
	{
		arrNodes.push_back((a[numElem] + a[numElem + 1]) / 2.0);
		Create_Function(dimTask, a, arrNodes[0], numElem, D, ResMat);
		ResMat = ResMat * h;
	}
	else if (Type_Integration == "Trapezoidal_Type")
	{
		arrNodes.push_back(a[numElem]);
		arrNodes.push_back(a[numElem + 1]);

		MatrixSchwarz M1;
		MatrixSchwarz M2;

		Create_Function(dimTask, a, arrNodes[0], numElem, D, ResMat);
		Create_Function(dimTask, a, arrNodes[1], numElem, D, ResMat);

		ResMat = M1 + M2;
		ResMat = ResMat * (h / 2.0);
	}
	else if (Type_Integration == "Gauss_2_Type")
	{

		arrNodes.push_back((a[numElem] + a[numElem + 1]) / 2.0 - (a[numElem + 1] - a[numElem]) / (2 * sqrt(3.0)));
		arrNodes.push_back((a[numElem] + a[numElem + 1]) / 2.0 + (a[numElem + 1] - a[numElem]) / (2 * sqrt(3.0)));

		MatrixSchwarz M1;
		MatrixSchwarz M2;

		Create_Function(dimTask, a, arrNodes[0], numElem, D, ResMat);
		Create_Function(dimTask, a, arrNodes[1], numElem, D, ResMat);

		ResMat = M1 + M2;
		ResMat = ResMat * (h / 2.0);
	}
	else
	{
		printf("You entered the wrong type of numerical integration: %s\n", Type_Integration.c_str());
		exit(0);
	}
}