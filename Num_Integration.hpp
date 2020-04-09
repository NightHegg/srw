
#include <cmath>
#include <vector>
#include <string>
#include "Classes_Schwarz.hpp"

using namespace std;

void Create_Function(double Node,
					 Basis_Functions &BasicElement,
					 MatrixSchwarz &D,
					 MatrixSchwarz &ResM)
{
	MatrixSchwarz B;
	MatrixSchwarz BTD;
	MatrixSchwarz BT;
	B.ConstructFullB(BasicElement, Node);
	B.Transpose(BT);
	BTD = BT * D;
	ResM = BTD * B;
	ResM = ResM * Node;
}

void Numerical_Integration(int Step,
						   VectorSchwarz &rr,
						   MatrixSchwarz &D,
						   Basis_Functions &ElementB,
						   string Type_Integration,
						   MatrixSchwarz &ResMat)
{
	double h = rr[1] - rr[0];
	std::vector<double> Node;
	vector<MatrixSchwarz *> SpecM;
	if (Type_Integration == "Riemann_Type")
	{
		Node.push_back((rr[Step] + rr[Step + 1]) / 2.0);
		Create_Function(Node[0], ElementB, D, ResMat);
		ResMat = ResMat * h;
	}
	else if (Type_Integration == "Trapezoidal_Type")
	{
		Node.push_back(rr[Step]);
		Node.push_back(rr[Step + 1]);
		
		MatrixSchwarz M1;
		MatrixSchwarz M2;

		Create_Function(Node[0], ElementB, D, M1);
		Create_Function(Node[1], ElementB, D, M2);

		ResMat = M1 + M2;
		ResMat = ResMat * (h / 2.0);
	}
	else if (Type_Integration == "Gauss_2_Type")
	{

		Node.push_back((rr[Step] + rr[Step + 1]) / 2.0 - (rr[Step + 1] - rr[Step]) / (2 * sqrt(3.0)));
		Node.push_back((rr[Step] + rr[Step + 1]) / 2.0 + (rr[Step + 1] - rr[Step]) / (2 * sqrt(3.0)));

		MatrixSchwarz M1;
		MatrixSchwarz M2;

		Create_Function(Node[0], ElementB, D, M1);
		Create_Function(Node[1], ElementB, D, M2);

		ResMat = M1 + M2;
		ResMat = ResMat * (h / 2.0);

	}
	else
	{
		printf("You entered the wrong type of numerical integration: %s\n",Type_Integration.c_str());
		exit(0);
	}
}