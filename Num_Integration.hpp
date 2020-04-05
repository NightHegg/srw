
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
	vector<MatrixSchwarz> SpecM;
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
	/*else
	if (Type_Integration == "Gauss_2_Type")
	{

		Node[0] = (rr[z] + rr[z + 1]) / 2.0 - (rr[z + 1] - rr[z]) / (2 * sqrt(3.0));
		Node[1]= (rr[z] + rr[z + 1]) / 2.0 + (rr[z + 1] - rr[z]) / (2 * sqrt(3.0));

		Create_Function(Type_Function, Node[0], z, h, rr, D, m, EpsC, Array_Dimensions, Matrix_1);
		Create_Function(Type_Function, Node[1], z, h, rr, D, m, EpsC, Array_Dimensions, Matrix_2);

		Sum_Matrix(Matrix_1, Matrix_2, Final_Matrix, i, j);
		Multiplying_Matrix(Final_Matrix, h / 2.0, i, j);

	}
	else
	if (Type_Integration == "Gauss_3_Type")
	{
		Node[0] = (rr[z] + rr[z + 1]) / 2.0 - (rr[z + 1] - rr[z])*sqrt(3.0 / 5.0) / (2.0);
		Node[1] = (rr[z] + rr[z + 1]) / 2.0 + (rr[z + 1] - rr[z])*sqrt(3.0 / 5.0) / (2.0);
		Node[2] = (rr[z] + rr[z + 1]) / 2.0;

		Create_Function(Type_Function, Node[0], z, h, rr, D, m, EpsC, Array_Dimensions,Matrix_1);
		Create_Function(Type_Function, Node[1], z, h, rr, D, m, EpsC, Array_Dimensions,Matrix_2);
		Create_Function(Type_Function, Node[2], z, h, rr, D, m, EpsC, Array_Dimensions,Matrix_3);

		Sum_Matrix(Matrix_1, Matrix_2, Interim_Matrix_1, i, j);
		Multiplying_Matrix(Interim_Matrix_1, 5.0*h / 18.0, i, j);
		Multiplying_Matrix(Matrix_3, 8.0*h / 18.0, i, j);
		Sum_Matrix(Interim_Matrix_1, Matrix_3, Final_Matrix, i, j);
	}
	else
	if (Type_Integration == "Simpson_Type")
	{
		Node[0] = rr[z];
		Node[1] = rr[z+1];
		Node[2] = (rr[z] + rr[z + 1]) / 2.0;

		Create_Function(Type_Function, Node[0], z, h, rr, D, m, EpsC, Array_Dimensions, Matrix_1);
		Create_Function(Type_Function, Node[1], z, h, rr, D, m, EpsC, Array_Dimensions, Matrix_2);
		Create_Function(Type_Function, Node[2], z, h, rr, D, m, EpsC, Array_Dimensions, Matrix_3);

		Sum_Matrix(Matrix_1, Matrix_2, Interim_Matrix_1, i, j);
		Multiplying_Matrix(Matrix_3, 4, i, j);
		Sum_Matrix(Interim_Matrix_1, Matrix_3, Final_Matrix, i, j);
		Multiplying_Matrix(Final_Matrix, h / 6.0, i, j);
	}
	else
	if (Type_Integration == "3_8_Type")
	{
		Node[0] = rr[z];
		Node[1] = rr[z + 1];
		Node[2] = (2 * rr[z] + rr[z + 1]) / 3.0;
		Node[3] = (rr[z] + 2 * rr[z + 1]) / 3.0;

		Create_Function(Type_Function, Node[0], z, h, rr, D, m, EpsC, Array_Dimensions, Matrix_1);
		Create_Function(Type_Function, Node[1], z, h, rr, D, m, EpsC, Array_Dimensions, Matrix_2);
		Create_Function(Type_Function, Node[2], z, h, rr, D, m, EpsC, Array_Dimensions, Matrix_3);
		Create_Function(Type_Function, Node[2], z, h, rr, D, m, EpsC, Array_Dimensions, Matrix_4);

		Sum_Matrix(Matrix_1, Matrix_2, Interim_Matrix_1, i, j);
		Sum_Matrix(Matrix_3, Matrix_4, Interim_Matrix_2, i, j);
		Multiplying_Matrix(Interim_Matrix_2, 3, i, j);
		Sum_Matrix(Interim_Matrix_1, Interim_Matrix_2, Final_Matrix, i, j);
		Multiplying_Matrix(Final_Matrix, h / 8.0, i, j);
	}*/
	else
	{
		cout << "���� ������� ������������ �������� ��� ���������� ��������������: " << Type_Integration << endl;
		exit(0);
	}
}