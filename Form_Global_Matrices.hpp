#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include "classes/Basis_Functions.hpp"
#include "classes/Strain_Matrix.hpp"
#include "Classes_Schwarz.hpp"
#include "Num_Integration.hpp"
#include "Methods.hpp"

using namespace std;

/*void formElemMatStiffness(int i, 
                            MatrixSchwarz& K, 
                            VectorSchwarz& rVec, 
                            int amntNodes,
                            MatrixSchwarz& D,
                            std::string typeIntegration)
{
    MatrixSchwarz localKM;
    Basis_Functions ElementB(i, rVec, amntNodes);
    Numerical_Integration(i, rVec, D, ElementB, typeIntegration, localKM);
    for (int j = 0; j < iB; j++)
    {
        for (int k = 0; k < jB; k++)
        {
            K[i + j][i + k] += localKM[j][k];
        }
    }
}*/

void Form_Glob_Mat_Stiffness(int dimTask, MatrixSchwarz &K, MatrixSchwarz &Ke, int numElem, vector<int> &localElements)
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
						K[localElements[j] * dimTask + i][localElements[k] * dimTask + n] += Ke[j * dimTask + i][k * dimTask + n];
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
							 vector<int> &localElements)
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

		for (auto x : localElements)
		{
			for (int i = 0; i < 2; i++)
				localNodes.push_back(mesh.GetElement(x * 2 + i));
		}

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

void Form_Boundary_Conditions(int dimTask, vector<double> &arrBound, VectorSchwarz &y, VectorSchwarz &mesh, MatrixSchwarz &K, VectorSchwarz &F)
{
	int Coef{0};
	double tmp{0};
	int size = mesh.GetSize();
	vector<double> localNodes;

	switch (dimTask)
	{
	case 1:
		F[0] += arrBound[2] * mesh[0];
		F[size - 1] += (-1.0) * arrBound[3] * mesh[size - 1];
		if (arrBound[0] != -1)
		{
			F[1] = F[1] - K[1][0] * arrBound[0];
			K[0][0] = 1;
			K[0][1] = 0;
			K[1][0] = 0;
			F[0] = arrBound[0];
		}
		if (arrBound[1] != -1)
		{
			F[size - 2] = F[size - 2] - K[size - 2][size - 1] * arrBound[size - 1];
			K[size - 1][size - 1] = 1;
			K[size - 1][size - 2] = 0;
			K[size - 2][size - 1] = 0;
			F[size - 1] = arrBound[size - 1];
		}
		break;
	case 2:
		ifstream scan("files/" + to_string(dimTask) + "D/nodes.dat");
		while (!scan.eof())
		{
			scan >> tmp;
			localNodes.push_back(tmp);
		}
		for (int i; i < 4; i++)
		{
			if ((i == 0 || i == 2) && (arrBound[i] != -1))
			{
				Coef == 1;
				for (int j = 0; j < mesh.GetSize() / 2; j++)
				{
					if (mesh.GetElement(j + Coef) == localNodes[i + Coef])
					{
						K[j * dimTask + Coef][j * dimTask + Coef] = 1;
						F[j * dimTask + Coef] = arrBound[i];
						for (int k; k < mesh.GetSize() / 2; k++)
						{
							if (k != j)
							{
								K[j * dimTask + Coef][k * dimTask + Coef] = 0;
								F[k * dimTask + Coef] = F[k * dimTask + Coef] - K[k * dimTask + Coef][j * dimTask + Coef] * arrBound[i];
								K[k * dimTask + Coef][j * dimTask + Coef] = 0;
							}
						}
					}
				}
			}
			if ((i == 1 || i == 3) && (arrBound[i] != -1))
			{
				Coef == 0;
				for (int j = 0; j < mesh.GetSize() / 2; j++)
				{
					if (mesh.GetElement(j + Coef) == localNodes[i + Coef])
					{
						K[j * dimTask + Coef][j * dimTask + Coef] = 1;
						F[j * dimTask + Coef] = arrBound[i];
						for (int k; k < mesh.GetSize() / 2; k++)
						{
							if (k != j)
							{
								K[j * dimTask + Coef][k * dimTask + Coef] = 0;
								F[k * dimTask + Coef] = F[k * dimTask + Coef] - K[k * dimTask + Coef][j * dimTask + Coef] * arrBound[i];
								K[k * dimTask + Coef][j * dimTask + Coef] = 0;
							}
						}
					}
				}
			}
		}
		for(int i=4;i<arrBound.size();i++)
		{
			
		}
		break;
	}
}

void Form_Boundary_Conditions_Schwarz(int dimTask,
									  vector<double> &arrBound,
									  VectorSchwarz &y,
									  VectorSchwarz &ySubd,
									  VectorSchwarz &yPreviousSubd,
									  VectorSchwarz &mesh, MatrixSchwarz &K,
									  VectorSchwarz &F)
{
	int size = ySubd.GetSize();

	switch (dimTask)
	{
	case 1:
		if (ySubd.Compare_Boundary_Left(y))
		{
			F[size - 2] = F[size - 2] - K[size - 2][size - 1] * yPreviousSubd[size - 1];
			K[size - 1][size - 1] = 1;
			K[size - 1][size - 2] = 0;
			K[size - 2][size - 1] = 0;
			F[size - 1] = yPreviousSubd[size - 1] * 1.0;

			F[0] += arrBound[2] * mesh[0];
		}
		else if (ySubd.Compare_Boundary_Right(y))
		{
			F[1] = F[1] - K[1][0] * ySubd[0];
			K[0][0] = 1;
			K[0][1] = 0;
			K[1][0] = 0;
			F[0] = ySubd[0];

			F[size - 1] += (-1.0) * arrBound[3] * mesh[size - 1];
		}
		else
		{
			F[1] = F[1] - K[1][0] * ySubd[0];
			K[0][0] = 1;
			K[0][1] = 0;
			K[1][0] = 0;
			F[0] = ySubd[0];

			F[size - 2] = F[size - 2] - K[size - 2][size - 1] * yPreviousSubd[size - 1];
			K[size - 1][size - 1] = 1;
			K[size - 1][size - 2] = 0;
			K[size - 2][size - 1] = 0;
			F[size - 1] = yPreviousSubd[size - 1] * 1.0;
		}
		break;
	}
}

void Form_Local_Element(VectorSchwarz &elements, vector<int> &localElements, int numElement)
{
	for (int i = 0; i < 3; i++)
		localElements.push_back(elements.GetElement(numElement * 3 + i));
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
			Form_Local_Element(elements, localElements, i);

		Form_Elem_Mat_Stiffness(dimTask, Ke, mesh, elements, D, S, i, amntNodes, localElements);
		Form_Glob_Mat_Stiffness(dimTask, K, Ke, i, localElements);
		Form_Elem_Vec_Right(Fe, i);
		Form_Glob_Vec_Right(F, Fe, i);
		localElements.clear();
	}
}

void Get_Displacements(int dimTask,
					   string *Route,
					   VectorSchwarz &y,
					   VectorSchwarz &mesh,
					   VectorSchwarz &elements,
					   strainMatrix &S,
					   MatrixSchwarz &D,
					   int uk,
					   int amntSubdomains,
					   double stopCriteria,
					   int amntNodes,
					   int amntElements)
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
		K.Construct(amntNodes * dimTask, amntNodes * dimTask);
		F.Construct(amntNodes * dimTask);
		*Route += "Non_Schwarz/";
		Ensembling(dimTask, K, F, D, S, mesh, elements, amntNodes, amntElements);
		Form_Boundary_Conditions(dimTask, arrBound, y, mesh, K, F);
		Tridiogonal_Algorithm_Right(amntNodes, K, F, y);
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

				Tridiogonal_Algorithm_Right(amntNodes, K, F, ySubd);

				y.ReturnAllocatedArrayResults(ySubd, i);

				K.~MatrixSchwarz();
				F.~VectorSchwarz();
			}
			Counter++;
		} while (y.ConvergenceL2(yPrevious, mesh) > stopCriteria);
		printf("Amount of iterations: %d\n\n", Counter);
	}
	y.SetName("y");

	y.Record(*Route, amntSubdomains, uk);

	Record_AddData(mesh.GetSize(), *Route, amntSubdomains, Counter, stopCriteria, coefOverlap);
}