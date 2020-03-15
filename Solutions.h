#pragma once
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include "omp.h"
#include "Methods.h"
#include "Classes.h"
#include "Num_Integration.h"

using namespace std;


void Progonka_Solution(int SchwarzStep,
	Vector& rr,
	double pa,
	double pb,
	Vector& y,
	Vector& yPrevious,
	Matrix& D,
	int TaskDim,
	int TaskAmNodes)
{
	setlocale(LC_ALL, "Russian");
	int iB = D.jM;
	int jB = TaskDim * TaskAmNodes;
	Vector yChosen;
	Vector rrChosen;
	Vector yPreviousChosen;
	if (y.UsingMethodSchwarz)
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
	Matrix KM(yChosen.iV, yChosen.iV);
	Vector F(yChosen.iV);
	Matrix A;
	string Type_Integration = "Riemann_Type";
	for (int i = 0; i < yChosen.iV - 1; i++)
	{
		BasicElements ElementB(i, rrChosen, TaskAmNodes);
		Numerical_Integration(i, rrChosen, D, ElementB, Type_Integration, A);
		for (int j = 0; j < iB; j++)
		{
			for (int k = 0; k < jB; k++)
			{
				KM[i + j][i + k] += A[j][k];
			}
		}
	}
	if (y.UsingMethodSchwarz)
	{
		if (yChosen.LeftBoundary == y.LeftBoundary)
		{
			F[yChosen.iV - 2] = F[yChosen.iV - 2] - KM[yChosen.iV - 2][yChosen.iV - 1] * yPreviousChosen[yChosen.iV - 1];
			KM[yChosen.iV - 1][yChosen.iV - 1] = 1;
			KM[yChosen.iV - 1][yChosen.iV - 2] = 0;
			KM[yChosen.iV - 2][yChosen.iV - 1] = 0;
			F[yChosen.iV - 1] = yPreviousChosen[yChosen.iV - 1] * 1.0;

			F[0] = pa * rrChosen[0];
		}
		else if (yChosen.RightBoundary == y.RightBoundary)
		{
			F[1] = F[1] - KM[1][0] * yChosen[0];
			KM[0][0] = 1;
			KM[0][1] = 0;
			KM[1][0] = 0;
			F[0] = yChosen[0];

			F[yChosen.iV - 1] = -pb * rrChosen[yChosen.iV - 1] * 1.0;
		}
		else
		{
			F[1] = F[1] - KM[1][0] * yChosen[0];
			KM[0][0] = 1;
			KM[0][1] = 0;
			KM[1][0] = 0;
			F[0] = yChosen[0];

			F[yChosen.iV - 2] = F[yChosen.iV - 2] - KM[yChosen.iV - 2][yChosen.iV - 1] * yPreviousChosen[yChosen.iV - 1];
			KM[yChosen.iV - 1][yChosen.iV - 1] = 1;
			KM[yChosen.iV - 1][yChosen.iV - 2] = 0;
			KM[yChosen.iV - 2][yChosen.iV - 1] = 0;
			F[yChosen.iV - 1] = yPreviousChosen[yChosen.iV - 1] * 1.0;
		}
	}
	else
	{
		F[0] += pa * rrChosen[0];
		F[yChosen.iV - 1] += -pb * rrChosen[yChosen.iV - 1];
	}
	Progonka_Method(yChosen.iV, KM, F, yChosen);
	if (y.UsingMethodSchwarz)
	{
		y.ReturnAllocatedArrayResults(yChosen, SchwarzStep);
	}
	else
	{
		y = yChosen;
	}
}