#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include "Num_Integration.hpp"
#include "Methods.hpp"
#include "classes.hpp"
#include "record_data.hpp"

using namespace std;

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

void FormArray_localElement(VectorSchwarz &elements, vector<int> &localElement, int numElement)
{
	for (int i = 0; i < 3; i++)
		localElement.push_back(elements.GetElement(numElement * 3 + i));
}
void FormArray_localMesh(vector<int> &localElement, VectorSchwarz &mesh, vector<double> &localMesh)
{
	for (auto x : localElement)
	{
		for (int i = 0; i < 2; i++)
			localMesh.push_back(mesh.GetElement(x * 2 + i));
	}
}

void Solve_Linear_System(int dimTask, MatrixSchwarz &K, VectorSchwarz &F, VectorSchwarz &y)
{
	switch (dimTask)
	{
	case 1:
		Tridiogonal_Algorithm_Right(K, F, y);
		break;
	case 2:
		Gaussian_Elimination(K, F, y);
		break;
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
		FormArray_localMesh(localElement, mesh, localNodes);

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

void Form_Boundary_Conditions(int dimTask, vector<double> &arrBound, VectorSchwarz &y, VectorSchwarz &mesh, VectorSchwarz &elements, int amntElements, int amntNodes, MatrixSchwarz &K, VectorSchwarz &F)
{
	bool keyDirichlet{false};
	bool keyNeumann{false};
	int Coef{0};
	double tmp{0};
	int size = mesh.GetSize();
	vector<double> coordsBound;
	vector<double> localMesh;
	vector<int> localElement;
	double A{0};

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
			coordsBound.push_back(tmp);
		}
		for (int i = 0; i < 4; i++)
		{
			if ((i == 0 || i == 2) && (arrBound[i] != -1))
			{
				Coef = 1;
				keyDirichlet = true;
			}
			if ((i == 0 || i == 2) && (arrBound[i + 4] != -1))
			{
				Coef = 1;
				keyNeumann = true;
			}
			if ((i == 1 || i == 3) && (arrBound[i] != -1))
			{
				Coef = 0;
				keyDirichlet = true;
			}
			if ((i == 1 || i == 3) && (arrBound[i + 4] != -1))
			{
				Coef = 0;
				keyNeumann = true;
			}
			if (keyDirichlet)
			{
				for (int j = 0; j < mesh.GetSize() / 2; j++)
				{
					if (mesh.GetElement(j * dimTask + Coef) == coordsBound[i * dimTask + Coef])
					{
						for (int k = 0; k < mesh.GetSize(); k++)
							K[j * dimTask + Coef][k] = 0;
						K[j * dimTask + Coef][j * dimTask + Coef] = 1;
						F[j * dimTask + Coef] = arrBound[i];
						for (int k = 0; k < mesh.GetSize(); k++)
						{
							if (k != j * dimTask + Coef)
							{
								F[k] = F[k] - K[k][j * dimTask + Coef] * arrBound[i];
								K[k][j * dimTask + Coef] = 0;
							}
						}
					}
				}
				keyDirichlet = false;
			}

			if (keyNeumann)
			{
				for (int j = 0; j < amntElements; j++)
				{
					FormArray_localElement(elements, localElement, j);
					FormArray_localMesh(localElement, mesh, localMesh);
					for (int k = 0; k < localMesh.size() / dimTask; k++)
					{
						if (localMesh[k * dimTask + Coef] == coordsBound[i * dimTask + Coef])
						{
							A=GetAreaTriangle(localMesh);
							F[j * dimTask + Coef] = F[j * dimTask + Coef] + (arrBound[i + 4] / 2.0) *A/3;
						}
					}
				}
				keyNeumann = false;
			}
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
			FormArray_localElement(elements, localElements, i);

		Form_Elem_Mat_Stiffness(dimTask, Ke, mesh, elements, D, S, i, amntNodes, localElements);
		Form_Glob_Mat_Stiffness(dimTask, K, Ke, i, localElements);
		Form_Elem_Vec_Right(Fe, i);
		Form_Glob_Vec_Right(F, Fe, i);
		localElements.clear();
	}
}

void CalcDisplacements(int dimTask,
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
		*Route += "Non_Schwarz/";
		K.Construct(amntNodes * dimTask, amntNodes * dimTask);
		F.Construct(amntNodes * dimTask);

		Ensembling(dimTask, K, F, D, S, mesh, elements, amntNodes, amntElements);

		Form_Boundary_Conditions(dimTask, arrBound, y, mesh, elements, amntElements, amntNodes, K, F);
		//K.SetName("K");
		//F.SetName("F");
		//K.Record("",1);
		//F.Record("",1);
		Solve_Linear_System(dimTask, K, F, y);
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

				Solve_Linear_System(dimTask, K, F, y);

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