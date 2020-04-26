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

void Solve(int N, int dimTask)
{
	int amntSubdomains;
	double stopCriteria;
	
	ifstream sch("files/"+to_string(dimTask)+"D/schwarz.dat");
	sch>>amntSubdomains;
	sch>>stopCriteria;

	int dimEps, dimSigma;
	switch (dimTask)
	{
		case 1:
		{
			dimEps=2;
			dimSigma=2;
			break;
		}
		case 2:
		{
			dimEps=3;
			dimSigma=3;
		}
	}
	MatrixSchwarz D(dimSigma, dimEps);
	MatrixSchwarz Eps(dimEps, N);
	MatrixSchwarz Sigma(dimSigma, N);

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
	double stopCriteria{1e-6};

	std::stringstream ss;
	ss<<stopCriteria;
	ss.precision(7);
	std::string sStopCriteria = ss.str();

	double lambda = (nyu * E) / ((1 + nyu) * (1 - 2 * nyu) * 1.0);
	double myu = E / (2 * (1 + nyu));

	int dimTask, amntNodes, dimEps, dimSigma;


	
	D.Elastic_Modulus_Tensor(dimTask);

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
			for (int i = 0; i < amntSubdomains; i++)
			{
				Progonka_Solution(i, rr, pa, pb, y, yPrevious, D, DimTask, AmNodes);
			}
			cout<<y.ConvergenceL2(yPrevious, rr)<<endl;
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