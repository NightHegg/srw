#pragma once
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include "Classes.h"
#include "omp.h"

void Create_Function(double Node, BasicElements& BasicElement, Matrix& D, Matrix& ResM)
{
	Matrix B;
	Matrix BTD;
	Matrix BT;
	B.ConstructFullB(BasicElement, Node);
	B.Transpose(BT);
	BTD = BT * D;
	ResM = BTD * B;
	ResM = ResM * Node;
}

void Numerical_Integration(int Step, Vector& rr, Matrix& D, BasicElements& ElementB, string Type_Integration, Matrix& ResMat)
{
	double h = rr[1] - rr[0];
	double* Node = new double[8];

	if (Type_Integration == "Riemann_Type")
	{
		Node[0] = (rr[Step] + rr[Step + 1]) / 2.0;
		Create_Function(Node[0], ElementB, D, ResMat);
		ResMat = ResMat * h; 

	}
	/*else
	if (Type_Integration == "Trapezoidal_Type")
	{
		Node[0] = rr[z];
		Node[1] = rr[z + 1];

		Create_Function(Type_Function, Node[0], z, h, rr, D, Array_Dimensions, Matrix_1);
		Create_Function(Type_Function, Node[1], z, h, rr, D, Array_Dimensions, Matrix_2);

		Sum_Matrix(Matrix_1, Matrix_2, Final_Matrix, i, j);
		Multiplying_Matrix(Final_Matrix, h / 2.0, i, j);
	}
	else
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