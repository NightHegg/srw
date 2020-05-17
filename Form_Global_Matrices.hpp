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

void Form_Glob_Mat_Stiffness(int dimTask, MatrixSchwarz &K, MatrixSchwarz &Ke, int numElem)
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
							 int amntNodes)
{
	string Type_Integration = "Trapezoidal_Type";
	vector<int> localElements;
	switch (dimTask)
	{
	case 1:
		Numerical_Integration(dimTask, numElement, mesh, elements, D, S, Type_Integration, Ke);
		break;
	case 2:
		for (int i = 0; i < dimTask; i++)
			localElements.push_back(elements.GetElement(numElement + i));
		break;
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
	int size = mesh.GetSize();

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

	for (int i = 0; i < amntElements; i++)
	{
		Form_Elem_Mat_Stiffness(dimTask, Ke, mesh, elements, D, S, i, amntNodes);
		Form_Glob_Mat_Stiffness(dimTask, K, Ke, i);
		Form_Elem_Vec_Right(Fe, i);
		Form_Glob_Vec_Right(F, Fe, i);
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
		K.Construct(amntNodes, amntNodes);
		F.Construct(amntNodes);
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