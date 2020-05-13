#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include "Classes_Schwarz.hpp"
#include "Solutions.hpp"
#include "Additional_Functions.h"
#include "Data_Record.h"
#include "classes/Strain_Matrix.hpp"
#include "FormGlobalMatrices.hpp"

using namespace std;

void Solve(int dimTask)
{
	string Route{"results/" + to_string(dimTask) + "D/"};
	int Counter{0};
	int amntNodes{0};
	int amntElements{0};
	int amntSubdomains;
	double stopCriteria;
	double coefOverlap{0};
	strainMatrix S(dimTask);

	ifstream sch("files/" + to_string(dimTask) + "D/schwarz.dat");
	sch >> amntSubdomains;
	sch >> stopCriteria;

	std::stringstream ss;
	ss << stopCriteria;
	ss.precision(7);
	std::string sStopCriteria = ss.str();

	int dimEps, dimSigma;

	double tmp;
	vector<double> tmpBuf;
	ifstream scanfN("files/" + to_string(dimTask) + "D/mesh.dat");
	while (!scanfN.eof())
	{
		scanfN >> tmp;
		tmpBuf.push_back(tmp);
		amntNodes++;
	}
	VectorSchwarz a(amntNodes);
	int tmpC{0};
	for (int x : tmpBuf)
		a.SetElement(tmpC, x);

	switch (dimTask)
	{
	case 1:
		dimEps = dimSigma = 2;
		amntElements = amntNodes - 1;
		break;
	case 2:
		dimEps = dimSigma = 3;
		ifstream scan("files/2D/elements.dat");
		while (!scan.eof())
			amntElements++;
		scan.close();
		break;
	default:
		printf("Wrong input: dimTask\n");
		break;
	}

	VectorSchwarz y(amntNodes);
	VectorSchwarz yPrevious(amntNodes);

	MatrixSchwarz Eps(dimEps, amntElements);
	MatrixSchwarz Sigma(dimSigma, amntElements);

	MatrixSchwarz D(dimSigma, dimEps);
	D.Elastic_Modulus_Tensor(dimTask);

	VectorSchwarz yChosen;
	VectorSchwarz aChosen;
	VectorSchwarz yPreviousChosen;

	if (amntSubdomains < 2)
		Get_Displacements(dimTask, y, yPrevious, a, D, S);
	else
	{
		y.Fill(-1e-6);
		Route += "Schwarz/SC_" + sStopCriteria + "/";
		a.Decomposition(amntSubdomains, &coefOverlap);
		y.Equal_SchwarzNodes(a);
		yPrevious.Equal_SchwarzNodes(a);
		do
		{
			yPrevious = y;
			for (int i = 0; i < amntSubdomains; i++)
			{
				yChosen = y.CreateAllocatedArray(i);
				aChosen = a.CreateAllocatedArray(i);
				yPreviousChosen = yPrevious.CreateAllocatedArray(i);
				Get_Displacements(dimTask, yChosen, yPreviousChosen, aChosen, D, S);
				y.ReturnAllocatedArrayResults(yChosen, i);
			}
		} while (y.ConvergenceL2(yPrevious, a) > stopCriteria);
	}

	/*double bufferValue{0};
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
	ss << stopCriteria;
	ss.precision(7);
	std::string sStopCriteria = ss.str();

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
			cout << y.ConvergenceL2(yPrevious, rr) << endl;
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
	Record_AddData(N, amntSubdomains, Counter, stopCriteria, coefOverlap, Route);*/
}