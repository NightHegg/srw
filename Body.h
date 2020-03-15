#pragma once
#include <vector>
#include "classes.h"
#include "solutions.h"
#include "additional_functions.h"

using namespace std;


void Solve(int N, int Amount_Subdomains)
{
	double Buffer_Value{ 0 };
	vector<double> Temporary_Buffer;
	ifstream ifs("F:\\YandexDisk\\University\\SRW\\NIR\\NIR\\dataFiles\\mainData.dat");
	while (!ifs.eof())
	{
		ifs >> Buffer_Value;
		Temporary_Buffer.push_back(Buffer_Value);
	}
	double a{ Temporary_Buffer.at(0) };
	double b{ Temporary_Buffer.at(1) };
	double pa{ Temporary_Buffer.at(2) };
	double pb{ Temporary_Buffer.at(3) };
	double E{ Temporary_Buffer.at(4) };
	double nyu{ Temporary_Buffer.at(5) };
	double uk{ Temporary_Buffer.at(6) };
	double rk{ Temporary_Buffer.at(7) };

	printf("%f\n", a);

	int DimTask = 1; // Dimension of the main task - 1 (1D), 2 (2D)
	int AmNodes = 2; // Amound of the nodes 
	int EpsDimArray = 2 * DimTask; //Size of the Eps array
	int SigmaDimArray = 2 * DimTask; //���������� ����������, � ������� �������� (1D) - ���������� � �������� ����������

	double myu = E / (2 * (1 + nyu));
	double lambda = (nyu*E) / ((1 + nyu)*(1 - 2 * nyu)*1.0);
	double K1 = lambda + 2 * myu; //���������� ��� ��������� Sigma
	double K2 = lambda; //���������� ��� ��������� Sigma
	Vector rr(N + 1);
	double h = (b - a) / (N*1.0); //���
	rr[0] = a; //��������� ������� ��� ������� �������� 
	for (int i = 1; i < N + 1; i++) //�������� ������� ��������
		rr[i] = rr[i - 1] + h;
	Vector y(N + 1);
	Vector yPrevious(N + 1);
	y.FillVector(-1e-6);
	Matrix D(SigmaDimArray, EpsDimArray);

	for (int i = 0; i < D.iM; i++) // ����� �������� ������� ������� D
	{
		for (int j = 0; j < D.jM; j++)
		{
			if (i == j)
				D[i][j] = lambda + 2 * myu;
			else
				D[i][j] = lambda;
		}
	}
	int Counter{ 0 };
	Matrix Eps(EpsDimArray, N);
	Matrix Sigma(SigmaDimArray, N);
	int SchwarzSteps;
	if (Amount_Subdomains < 2)
	{
		SchwarzSteps = 1;
		Progonka_Solution(SchwarzSteps, rr, pa, pb, y, yPrevious, D, DimTask, AmNodes);
	}
	else
	{
		rr.Decomposition(Amount_Subdomains);
		for (vector<int>::iterator it = rr.SchwarzNodes.begin();it != rr.SchwarzNodes.end();++it)
		{
			y.SchwarzNodes.push_back(*it);
			yPrevious.SchwarzNodes.push_back(*it);
			cout << *it << endl;
		}
		cout << endl;
		y.UsingMethodSchwarz = true;
		do
		{
			printf("Eps=%g\n", y.ConvergenceL2(yPrevious, rr));
			yPrevious = y;
			y.FillVector(0);
			for (int i = 0; i < Amount_Subdomains; i++)
			{
				Progonka_Solution(i, rr, pa, pb, y, yPrevious, D, DimTask, AmNodes);
			}
			Counter++;
		} while (y.ConvergenceL2(yPrevious, rr) > 1e-6);
	}
	printf("\n�������� ��������: %g\n���������� ��������: %d\n\n", y.ConvergenceL2(yPrevious, rr), Counter);
	y.Show();
	Get_Eps(rr, y, Eps);
	Get_Sigma(D, Eps, Sigma);
	Matrix SigmaT;
	Sigma.Transpose(SigmaT);
	//SigmaT.Show();
	//Sigma.Show();
	//Record_Important_Data(N, uk, rk, y, Sigma, Array_Names_Files, Number_Operation);
	//Record_Solution_Additional_Settings(end - start, N, Number_Operation, Array_Names_Files[3]);
}