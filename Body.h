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
#include "Form_Global_Matrices.hpp"

using namespace std;

void Solve(vector<double> data)
{
	int coefOverlap{0}, amntNodes{0}, uk, rk, tmpV;

	int dimTask = data.at(0);
	if (dimTask == 1)
	{
		amntNodes = data.at(1);
	}
	int amntSubdomains = data.at(2);
	if (amntSubdomains >= 2)
	{
		coefOverlap = data.at(3);
	}

	VectorSchwarz mesh, elements;

	string Route{"results/" + to_string(dimTask) + "D/"};

	double tmp, tmpCount{0};
	vector<double> tmpBuf;

	ifstream scanV("files/" + to_string(dimTask) + "D/coefs.dat");
	scanV >> tmpV;
	scanV >> uk;
	scanV >> rk;
	scanV.close();

	int amntElements{0}, amntNodes{0}, dimEps{0}, dimSigma{0};

	ifstream scan("files/" + to_string(dimTask) + "D/mesh.dat");
	while (!scan.eof())
	{
		scan >> tmp;
		tmpBuf.push_back(tmp);
		amntNodes++;
	}
	scan.close();

	mesh.Construct(amntNodes);
	for (double x : tmpBuf)
	{
		mesh.SetElement(tmpCount, x);
		tmpCount++;
	}

	tmpBuf.erase(tmpBuf.begin(), tmpBuf.end());
	tmpCount = 0;

	switch (dimTask)
	{
	case 1:
		dimEps = dimSigma = 2;
		amntElements = amntNodes - 1;
		elements.Construct(amntElements);
		break;
	case 2:
	{
		dimEps = dimSigma = 3;
		ifstream scan("files/2D/elements.dat");
		while (!scan.eof())
		{
			scan >> tmp;
			tmpBuf.push_back(tmp);
			amntElements++;
		}
		scan.close();
		break;
	}
	default:
		printf("Wrong input: dimTask\n");
		break;
	}
	VectorSchwarz y(amntNodes);
	MatrixSchwarz Eps(dimEps, amntElements);
	MatrixSchwarz Sigma(dimSigma, amntElements);

	strainMatrix S(dimTask, mesh);

	MatrixSchwarz D(dimSigma, dimEps);
	D.Elastic_Modulus_Tensor(dimTask);

	Get_Displacements(dimTask, &Route, y, mesh, elements, S, D, uk);
	Eps.Create_Sy(S, y);
	Sigma = D * Eps;

	Sigma.SetName("Sigma");

	Sigma.Record(Route, dimTask, rk);

	//
	//

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