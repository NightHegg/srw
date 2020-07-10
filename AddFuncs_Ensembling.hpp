#ifndef ADDFUNCS_ENSEMBLING_HPP
#define ADDFUNCS_ENSEMBLING_HPP

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include "Num_Integration.hpp"
#include "Methods.hpp"
#include "classes.hpp"
#include "record_data.hpp"

double GetAreaTriangle(vector<double> &localNodes)
{
	vector<double> a;
	vector<double> b;
	vector<double> c;

	a.push_back(localNodes[2] * localNodes[5] - localNodes[4] * localNodes[3]);
	a.push_back(localNodes[4] * localNodes[1] - localNodes[5] * localNodes[0]);
	a.push_back(localNodes[0] * localNodes[3] - localNodes[2] * localNodes[1]);

	b.push_back(localNodes[3] - localNodes[5]);
	b.push_back(localNodes[5] - localNodes[1]);
	b.push_back(localNodes[1] - localNodes[3]);

	c.push_back(localNodes[4] - localNodes[2]);
	c.push_back(localNodes[0] - localNodes[4]);
	c.push_back(localNodes[2] - localNodes[0]);

	double A = (1 / 2.0) * (localNodes[2] * localNodes[5] - localNodes[4] * localNodes[3] + localNodes[0] * localNodes[3] -
							localNodes[0] * localNodes[5] + localNodes[4] * localNodes[1] - localNodes[2] * localNodes[1]);
	return A;
}

void FormLocalElement(VectorSchwarz &elements, vector<int> &localElement, int numElement)
{
	for (int i = 0; i < 3; i++)
		localElement.push_back(elements.GetElement(numElement * 3 + i));
}
void FormLocalMesh(vector<int> &localElement, VectorSchwarz &mesh, vector<double> &localMesh)
{
	for (auto x : localElement)
	{
		for (int i = 0; i < 2; i++)
			localMesh.push_back(mesh.GetElement(x * 2 + i));
	}
}

void Form_Glob_Mat_Stiffness(int dimTask, MatrixSchwarz &K, MatrixSchwarz &Ke, int numElem, vector<int> &localElement)
{
	int size{0};
	int amntBF;
	switch (dimTask)
	{
	case 1:
		amntBF = 2;
		size = dimTask * amntBF;
		for (int j = 0; j < size; j++)
		{
			for (int k = 0; k < size; k++)
			{
				K[numElem + j][numElem + k] += Ke[j][k];
			}
		}
		break;
	case 2:
		amntBF = 3;
		size = dimTask * amntBF;
		for (int j = 0; j < amntBF; j++)
		{
			for (int k = 0; k < amntBF; k++)
			{
				for (int i = 0; i < dimTask; i++)
				{
					for (int n = 0; n < dimTask; n++)
					{
						K[localElement[j] * dimTask + i][localElement[k] * dimTask + n] += Ke[j * dimTask + i][k * dimTask + n];
					}
				}
			}
		}
		break;
	default:
		break;
	}
}

void Form_Elem_Mat_Stiffness(int dimTask,
							 MatrixSchwarz &Ke,
							 VectorSchwarz &mesh,
							 VectorSchwarz &elements,
							 MatrixSchwarz &D,
							 strainMatrix &S,
							 int numElement,
							 int amntNodes,
							 vector<int> &localElement)
{
	string Type_Integration = "Trapezoidal_Type";
	MatrixSchwarz B;
	MatrixSchwarz BT;
	MatrixSchwarz BTD;
	vector<double> localNodes;

	vector<double> a;
	vector<double> b;
	vector<double> c;
	double A{0};
	switch (dimTask)
	{
	case 1:
		Numerical_Integration(dimTask, numElement, mesh, elements, D, S, Type_Integration, Ke);
		break;
	case 2:
	{
		B.Construct(3, dimTask * 3);
		FormLocalMesh(localElement, mesh, localNodes);

		a.push_back(localNodes[2] * localNodes[5] - localNodes[4] * localNodes[3]);
		a.push_back(localNodes[4] * localNodes[1] - localNodes[5] * localNodes[0]);
		a.push_back(localNodes[0] * localNodes[3] - localNodes[2] * localNodes[1]);

		b.push_back(localNodes[3] - localNodes[5]);
		b.push_back(localNodes[5] - localNodes[1]);
		b.push_back(localNodes[1] - localNodes[3]);

		c.push_back(localNodes[4] - localNodes[2]);
		c.push_back(localNodes[0] - localNodes[4]);
		c.push_back(localNodes[2] - localNodes[0]);

		A = (1 / 2.0) * (localNodes[2] * localNodes[5] - localNodes[4] * localNodes[3] + localNodes[0] * localNodes[3] -
						 localNodes[0] * localNodes[5] + localNodes[4] * localNodes[1] - localNodes[2] * localNodes[1]);

		for (int j = 0; j < 3; j++)
		{
			B.SetElement(0, 2 * j, b[j] / (2 * A));
			B.SetElement(1, 1 + 2 * j, c[j] / (2 * A));
			B.SetElement(2, 2 * j, c[j] / (4 * A));
			B.SetElement(2, 1 + 2 * j, b[j] / (4 * A));
		}
		B.Transpose(BT);
		BTD = BT * D;
		Ke = BTD * B;
		Ke = Ke * A;
		a.clear();
		b.clear();
		c.clear();
		break;
	}
	}
}
void Form_Elem_Vec_Right(VectorSchwarz &Fe, int numElem)
{
}

void Form_Glob_Vec_Right(VectorSchwarz &F, VectorSchwarz &Fe, int numElem)
{
}

void Ensembling(int dimTask,
				MatrixSchwarz &K,
				VectorSchwarz &F,
				MatrixSchwarz &D,
				strainMatrix &S,
				VectorSchwarz &mesh,
				VectorSchwarz &elements,
				int amntNodes,
				int amntElements)
{
	int amntBE;
	switch (dimTask)
	{
	case 1:
		amntBE = 2;
		break;
	case 2:
		amntBE = 3;
		break;
	}

	MatrixSchwarz Ke(dimTask * amntBE, dimTask * amntBE);
	VectorSchwarz Fe(dimTask * amntBE);
	vector<int> localElements;
	for (int i = 0; i < amntElements; i++)
	{
		if (dimTask == 2)
			FormLocalElement(elements, localElements, i);

		Form_Elem_Mat_Stiffness(dimTask, Ke, mesh, elements, D, S, i, amntNodes, localElements);
		Form_Glob_Mat_Stiffness(dimTask, K, Ke, i, localElements);
		Form_Elem_Vec_Right(Fe, i);
		Form_Glob_Vec_Right(F, Fe, i);
		localElements.clear();
	}
}

#endif