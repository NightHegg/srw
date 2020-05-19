#ifndef METHODS_HPP
#define METHODS_HPP

#include <cmath>
#include <string>
#include <iostream>
#include "Classes_Schwarz.hpp"

using namespace std;

void Tridiogonal_Algorithm_Right(MatrixSchwarz &A, VectorSchwarz &F, VectorSchwarz &y)
{
	int N = A.GetSize_i();
	double denom;
	double *Alpha = new double[N - 1];
	double *Beta = new double[N];
	Alpha[0] = (-A[0][1]) / (A[0][0]);
	Beta[0] = (F[0]) / (A[0][0]);

	for (int i = 1; i < N - 1; i++)
	{
		denom = A[i][i] + (A[i][i - 1] * Alpha[i - 1]);
		Alpha[i] = (-A[i][i + 1]) / (denom * 1.0);
	}
	for (int i = 1; i < N; i++)
	{
		denom = A[i][i] + (A[i][i - 1] * Alpha[i - 1]);
		Beta[i] = (F[i] - A[i][i - 1] * Beta[i - 1]) / (denom * 1.0);
	}
	y[N - 1] = Beta[N - 1];
	for (int i = N - 2; i >= 0; i--)
	{
		y[i] = (y[i + 1] * Alpha[i]) + Beta[i];
	}
}

void Tridiogonal_Algorithm_Left(MatrixSchwarz &A, VectorSchwarz &F, VectorSchwarz &y)
{
	int N = A.GetSize_i();
	double denom;
	double *Dzeta = new double[N];
	double *Eta = new double[N];

	Dzeta[N - 1] = (-A[N - 1][N - 2]) / (A[N - 1][N - 1]);
	Eta[N - 1] = F[N - 1] / (A[N - 1][N - 1]);

	for (int i = N - 2; i > 0; i--)
	{
		denom = A[i][i] + (Dzeta[i + 1] * A[i][i + 1]);
		Dzeta[i] = -A[i][i - 1] / (denom * 1.0);
	}
	for (int i = N - 2; i >= 0; i--)
	{
		denom = A[i][i] + (Dzeta[i + 1] * A[i][i + 1]);
		Eta[i] = (F[i] - Eta[i + 1] * A[i][i + 1]) / (denom * 1.0);
	}
	y[0] = Eta[0];
	for (int i = 1; i < N; i++)
	{
		y[i] = (Dzeta[i] * y[i - 1]) + Eta[i];
	}
}

void Gaussian_Elimination(MatrixSchwarz &A, VectorSchwarz &F, VectorSchwarz &y)
{
	int dimTask{2};
	double buf{0}, sum{0};
	int N = A.GetSize_i() / dimTask;
	for (int i = 0; i < N; i++)
	{
		for (int k = 0; k < dimTask; k++)
		{
			buf = A[i * dimTask + k][i * dimTask + k];
			for (int j = 0; j < N; j++)
			{
				A[i * dimTask + k][j * dimTask + k] /= buf;
			}
			F[i * dimTask + k] /= buf;
		}
		for (int k = i + 1; k < N; k++)
		{
			for (int l = 0; l < dimTask; l++)
			{
				buf = A[k * dimTask + l][i * dimTask + l] * 1.0;
				for (int j = i; j < N; j++)
				{
					A[k * dimTask + l][j * dimTask + l] -= A[i * dimTask + l][j * dimTask + l] * buf * 1.0;
				}
				F[k * dimTask + l] -= F[i * dimTask + i] * buf * 1.0;
			}
		}
	}
	//A.Show();
	//F.Show();
	y[N*dimTask-1]=F[N*dimTask-1];
	y[N*dimTask-2]=F[N*dimTask-2];
	for (int i = N-2; i >= 0; i--)
	{
		for (int k = 0; k < dimTask; k++)
		{
			for (int j = i + 1; j < N; j++)
			{
				sum += A[i*dimTask+k][j*dimTask+k] * y[j*dimTask+k];
			}
			y[i*dimTask+k] = F[i*dimTask+k] - sum;
			sum = 0;
		}
	}
}

#endif