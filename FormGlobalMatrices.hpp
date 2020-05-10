#include <iostream>
#include <string>
#include "classes/Basis_Functions.hpp"
#include "classes/Strain_Matrix.hpp"
#include "Classes_Schwarz.hpp"
#include "Num_Integration.hpp"
#include "Methods.hpp"

/*void formElemMatStiffness(int i, 
                            MatrixSchwarz& KM, 
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
            KM[i + j][i + k] += localKM[j][k];
        }
    }
}*/

void Form_Elem_Mat_Stiffness(int dimTask, MatrixSchwarz Ke, VectorSchwarz &a, MatrixSchwarz &D, int numElem)
{
    string Type_Integration = "Trapezoidal_Type";
    //Basis_Functions ElementB(numElem, a);
    Numerical_Integration(dimTask, numElem, a, D, Type_Integration, Ke);
}

void Ensembling(int dimTask, MatrixSchwarz &K, VectorSchwarz &F, MatrixSchwarz &D, VectorSchwarz &a, int amntElements)
{
    MatrixSchwarz Ke(D.GetSize_i(), D.GetSize_i());
    VectorSchwarz Fe(D.GetSize_i());

    for (int i = 0; i < amntElements; i++)
    {
        Form_Elem_Mat_Stiffness(dimTask, Ke, a, D, i);
        Form_Glob_Mat_Stiffness(K, Ke, i);
        Form_Elem_Vec_Right(Fe, i);
        Form_Glob_Vec_Right(F, Fe, i);
        Form_Boundary_Conditions(K, F);
    }
}

void Get_Displacements(int dimTask, VectorSchwarz &y, VectorSchwarz &yPrevious, VectorSchwarz &a, MatrixSchwarz &D)
{
    int amntNodes = y.GetSize();
    int amntElements;
    switch (dimTask)
    {
    case 1:
    {
        amntElements = amntNodes - 1;
    }
    case 2:
    {
        ifstream scan("files/2D/elements.dat");
        while (!scan.eof())
        {
            amntElements++;
        }
        scan.close();
    }
    }

    MatrixSchwarz K(amntNodes, amntNodes);
    VectorSchwarz F(amntNodes);

    Ensembling(dimTask, K, F, D, a, amntElements);
    Tridiogonal_Algorithm(amntNodes, K, F, y);
}

/*void Progonka_Solution(int SchwarzStep,
	VectorSchwarz& rr,
	double pa,
	double pb,
	VectorSchwarz& y,
	VectorSchwarz& yPrevious,
	MatrixSchwarz& D,
	int TaskDim,
	int TaskAmNodes)
{

	int iB = D.GetSize_i();
	int jB = TaskDim * TaskAmNodes;
	VectorSchwarz yChosen;
	VectorSchwarz rrChosen;
	VectorSchwarz yPreviousChosen;
	if (y.Condition_Schwarz())
	{
		yChosen = y.CreateAllocatedArray(SchwarzStep);
		rrChosen = rr.CreateAllocatedArray(SchwarzStep);
		yPreviousChosen = yPrevious.CreateAllocatedArray(SchwarzStep);
	}
	else
	{
		yChosen = y;
		rrChosen = rr;
		yPreviousChosen = yPrevious;
	}
	int SizeDomain=yChosen.GetSize();
	Matrix KM(SizeDomain, SizeDomain);
	Vector F(SizeDomain);
	MatrixSchwarz A;
	string Type_Integration = "Trapezoidal_Type";
	for (int i = 0; i < SizeDomain - 1; i++)
	{
		Basis_Functions ElementB(i, rrChosen, TaskAmNodes);
		Numerical_Integration(i, rrChosen, D, ElementB, Type_Integration, A);
		for (int j = 0; j < iB; j++)
		{
			for (int k = 0; k < jB; k++)
			{
				KM[i + j][i + k] += A[j][k];
			}
		}
	}
	if (y.Condition_Schwarz())
	{
		if (yChosen.Compare_Boundary_Left(y))
		{
			F[SizeDomain - 2] = F[SizeDomain - 2] - KM[SizeDomain - 2][SizeDomain - 1] * yPreviousChosen[SizeDomain - 1];
			KM[SizeDomain - 1][SizeDomain - 1] = 1;
			KM[SizeDomain - 1][SizeDomain - 2] = 0;
			KM[SizeDomain - 2][SizeDomain - 1] = 0;
			F[SizeDomain - 1] = yPreviousChosen[SizeDomain - 1] * 1.0;

			F[0] = pa * rrChosen[0];
		}
		else if (yChosen.Compare_Boundary_Right(y))
		{
			F[1] = F[1] - KM[1][0] * yChosen[0];
			KM[0][0] = 1;
			KM[0][1] = 0;
			KM[1][0] = 0;
			F[0] = yChosen[0];

			F[SizeDomain - 1] = -pb * rrChosen[SizeDomain - 1] * 1.0;
		}
		else
		{
			F[1] = F[1] - KM[1][0] * yChosen[0];
			KM[0][0] = 1;
			KM[0][1] = 0;
			KM[1][0] = 0;
			F[0] = yChosen[0];

			F[SizeDomain - 2] = F[SizeDomain - 2] - KM[SizeDomain - 2][SizeDomain - 1] * yPreviousChosen[SizeDomain - 1];
			KM[SizeDomain - 1][SizeDomain - 1] = 1;
			KM[SizeDomain - 1][SizeDomain - 2] = 0;
			KM[SizeDomain - 2][SizeDomain - 1] = 0;
			F[SizeDomain - 1] = yPreviousChosen[SizeDomain - 1] * 1.0;
		}
	}
	else
	{
		F[0] += pa * rrChosen[0];
		F[SizeDomain - 1] += -pb * rrChosen[SizeDomain - 1];
	}
	Tridiogonal_Algorithm(SizeDomain, KM, F, yChosen);
	if (y.Condition_Schwarz())
	{
		y.ReturnAllocatedArrayResults(yChosen, SchwarzStep);
	}
	else
	{
		y = yChosen;
	}

}*/