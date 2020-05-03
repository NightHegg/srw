
#include <iostream>
#include <cmath>
#include <string>
#include "Classes_Schwarz.hpp"
#include "Num_Integration.hpp"
#include "Methods.hpp"

using namespace std;

void Progonka_Solution(int SchwarzStep,
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

}