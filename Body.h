
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include "Classes_Schwarz.hpp"
#include "Solutions.hpp"
#include "Additional_Functions.h"
#include "Data_Record.h"

using namespace std;

void Solve(int N, int amntSubdomains)
{
	double bufferValue{0};
	vector<double> tempBuffer;
	ifstream ifs("files/mainData.dat");
	while (!ifs.eof())
	{
		ifs >> bufferValue;
		tempBuffer.push_back(bufferValue);
	}
	double a{tempBuffer.at(0)};
	double b{tempBuffer.at(1)};
	double pa{tempBuffer.at(2)};
	double pb{tempBuffer.at(3)};
	double E{tempBuffer.at(4)};
	double nyu{tempBuffer.at(5)};
	double uk{tempBuffer.at(6)};
	double rk{tempBuffer.at(7)};

	double coefOverlap{0};
	string Route{"results/"};
	int Counter{0};
	Route += "1D/";
	double stopCriteria{1e-5};

	std::ostringstream ss;
	ss<<stopCriteria;
	std::string sStopCriteria = ss.str();

	double lambda = (nyu * E) / ((1 + nyu) * (1 - 2 * nyu) * 1.0);
	double myu = E / (2 * (1 + nyu));

	int DimTask = 1;				 // Dimension of the main task - 1 (1D), 2 (2D)
	int AmNodes = 2;				 // Amound of the nodes
	int EpsDimArray = 2 * DimTask;	 //Size of the Eps array
	int SigmaDimArray = 2 * DimTask; //Size of the Sigma array

	MatrixSchwarz D(SigmaDimArray, EpsDimArray);
	D.Elastic_Modulus_Tensor(lambda, myu);

	MatrixSchwarz Eps(EpsDimArray, N);
	MatrixSchwarz Sigma(SigmaDimArray, N);
	VectorSchwarz rr(N + 1);
	VectorSchwarz y(N + 1);
	VectorSchwarz yPrevious(N + 1);

	rr.Partition(a, b);
	y.Fill(-1e-6);

	if (amntSubdomains < 2)
	{
		Route += "Non_Schwarz/";
		Progonka_Solution(1, rr, pa, pb, y, yPrevious, D, DimTask, AmNodes);
	}
	else
	{
		Route += "Schwarz/SC_" + sStopCriteria + "/";
		rr.Decomposition(amntSubdomains, &coefOverlap);
		y.Equal_SchwarzNodes(rr);
		yPrevious.Equal_SchwarzNodes(rr);
		do
		{
			yPrevious = y;
			y.Fill(0);
			for (int i = 0; i < amntSubdomains; i++)
			{
				Progonka_Solution(i, rr, pa, pb, y, yPrevious, D, DimTask, AmNodes);
			}
			Counter++;
		} while (y.ConvergenceL2(yPrevious, rr) > stopCriteria);
		printf("\nThe stop criteria: %g\nAmount of iterations: %d\n\n", y.ConvergenceL2(yPrevious, rr), Counter);
	}
	Get_Eps(rr, y, Eps);
	Sigma = D * Eps;

	y.SetName("y");
	Sigma.SetName("Sigma");

	y.Record(Route, amntSubdomains, uk);
	Sigma.Record(Route, amntSubdomains, rk);
	Record_AddData(N, amntSubdomains, Counter, stopCriteria, coefOverlap, Route);
}