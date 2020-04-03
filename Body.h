
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "Classes_Schwarz.hpp"
#include "Solutions.hpp"
#include "Additional_Functions.h"
#include "Data_Record.h"

using namespace std;


void Solve(int N, int Amount_Subdomains)
{
	double Coef_Overflow{0};
	string Route{"results/"};
	int Counter{ 0 };
	Route+="1D/";
	double stopCriteria{1e-4};
	double Buffer_Value{ 0 };
	vector<double> Temporary_Buffer;
	ifstream ifs("files/mainData.dat");
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

	int DimTask = 1; // Dimension of the main task - 1 (1D), 2 (2D)
	int AmNodes = 2; // Amound of the nodes 
	int EpsDimArray = 2 * DimTask; //Size of the Eps array
	int SigmaDimArray = 2 * DimTask; //Size of the Sigma array
	
	double lambda = (nyu*E) / ((1 + nyu)*(1 - 2 * nyu)*1.0);
	double myu = E / (2 * (1 + nyu));
	VectorSchwarz rr(N + 1);
	double h = (b - a) / (N*1.0); 
	rr[0]=a;
	for (int i = 1; i < N + 1; i++) 
		rr[i]=rr[i-1]+h;
	VectorSchwarz y(N + 1);
	VectorSchwarz yPrevious(N + 1);
	y.Fill(-1e-6);
	MatrixSchwarz D(SigmaDimArray, EpsDimArray);
	D.Elastic_Modulus_Tensor(lambda, myu);
	MatrixSchwarz Eps(EpsDimArray, N);
	MatrixSchwarz Sigma(SigmaDimArray, N);
	if (Amount_Subdomains < 2)
	{
		Route+="Non_Schwarz/";
		Progonka_Solution(1, rr, pa, pb, y, yPrevious, D, DimTask, AmNodes);
	}
	else
	{
		Route+="Schwarz/";
		rr.Decomposition(Amount_Subdomains, &Coef_Overflow);
		y.Equal_SchwarzNodes(rr);
		yPrevious.Equal_SchwarzNodes(rr);
		do
		{
			yPrevious = y;
			y.Fill(0);
			for (int i = 0; i < Amount_Subdomains; i++)
			{
				Progonka_Solution(i, rr, pa, pb, y, yPrevious, D, DimTask, AmNodes);
			}
			Counter++;
		} while (y.ConvergenceL2(yPrevious, rr) > stopCriteria);
		printf("\nThe stop criteria: %g\nAmount of iterations: %d\n\n", y.ConvergenceL2(yPrevious, rr), Counter);
	}
	Get_Eps(rr, y, Eps);
	Sigma = D * Eps;
	Record_Results(y, Sigma,uk,rk, Amount_Subdomains, Route);
	Record_AddData(N, Amount_Subdomains, Counter, stopCriteria, Coef_Overflow, Route);
}