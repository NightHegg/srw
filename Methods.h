#pragma once
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>

using namespace std;

/*void Progonka_Method(int n, double *A, double *C, double *B, double *F, double *x)
{
	double denom;
	double *alpha = new double[n - 1];
	double *beta = new double[n - 1];
	alpha[0] = (-B[0]) / (C[0]);
	beta[0] = (F[0]) / (C[0]);
	for (int k = 1; k < n - 1; k++)
	{
		denom = C[k] + (A[k] * alpha[k - 1]);
		alpha[k] = (-B[k]) / (denom*1.0);
		beta[k] = (F[k] - A[k] * beta[k - 1]) / (denom*1.0);
	}
	x[n - 1] = (F[n - 1] - A[n - 1] * beta[n - 2]) / (C[n - 1] + A[n - 1] * alpha[n - 2]);
	for (int k = n - 2; k >= 0; k--)
	{
		x[k] = (x[k + 1] * alpha[k]) + beta[k];
	}
}*/

void Progonka_Method(int N, Matrix& A, Vector& F, Vector& y)
{
	double denom;
	double *Alpha = new double[N - 1];
	double *Beta = new double[N];
	double *Dzeta = new double[N];
	double *Eta = new double[N];
	Alpha[0] = (-A[0][1]) / (A[0][0]);
	Beta[0] = (F[0]) / (A[0][0]);

	for (int i = 1; i < N - 1; i++)
	{
		denom = A[i][i] + (A[i][i - 1] * Alpha[i - 1]);
		Alpha[i] = (-A[i][i + 1]) / (denom*1.0);
	}
	for (int i = 1; i < N; i++)
	{
		denom = A[i][i] + (A[i][i - 1] * Alpha[i - 1]);
		Beta[i] = (F[i] - A[i][i - 1] * Beta[i - 1]) / (denom*1.0);
	}

	Dzeta[N - 1] = (-A[N - 1][N - 2]) / (A[N - 1][N - 1]);
	Eta[N - 1] = F[N - 1] / (A[N - 1][N - 1]);

	for (int i = N - 2;i > 0;i--)
	{
		denom = A[i][i] + (Dzeta[i + 1] * A[i][i + 1]);
		Dzeta[i] = -A[i][i - 1] / (denom*1.0);
	}
	for (int i = N - 2;i >= 0;i--)
	{
		denom = A[i][i] + (Dzeta[i + 1] * A[i][i + 1]);
		Eta[i] = (F[i] - Eta[i + 1] * A[i][i + 1]) / (denom*1.0);
	}
	/*y[0] = Eta[0];
	for (int i = 1;i < N;i++)
	{
		y[i] = (Dzeta[i] * y[i - 1]) + Eta[i];
	}*/


	y[N - 1] = Beta[N - 1];
	for (int i = N - 2; i >= 0; i--)
	{
		y[i] = (y[i + 1] * Alpha[i]) + Beta[i];
	}
}