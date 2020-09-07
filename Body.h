#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>

#include "record_data.hpp"
#include "Form_Global_Matrices.hpp"
#include "Methods.hpp"

#include "classes.hpp"

using namespace std;
// TODO It need to be more comfortable

void Solve(vector<double> data)
{
	double stopCriteria{0}, uk, rk;
	int amntNodes{0}, amntSubdomains{0};

	double tmp, tmpCount{0};
	vector<double> tmpBuf;

	int dimTask = data.at(0);
	if (dimTask == 1)
	{
		amntNodes = data.at(1);
		amntSubdomains = data.at(2);
		if (amntSubdomains > 1)
		{
			stopCriteria = data.at(3);
		}
	}

	string sAmntNodes;
	if (amntNodes < 10)
	{
		sAmntNodes = "00" + to_string(amntNodes);
	}
	else if (amntNodes < 100)
	{
		sAmntNodes = "0" + to_string(amntNodes);
	}
	else
	{
		sAmntNodes = to_string(amntNodes);
	}

	if (dimTask == 1)
	{
		ifstream scan("files/" + to_string(dimTask) + "D/mesh_" + sAmntNodes + ".dat");
		while (!scan.eof())
		{
			scan >> tmp;
			tmpBuf.push_back(tmp);
		}
		scan.close();
		amntNodes++;
	}
	else
	{
		ifstream scan("files/" + to_string(dimTask) + "D/mesh" + ".dat");
		while (!scan.eof())
		{
			scan >> tmp;
			tmpBuf.push_back(tmp);
			scan >> tmp;
			tmpBuf.push_back(tmp);
			amntNodes++;
		}
		scan.close();
	}

	VectorSchwarz mesh, elements;

	string Route{"results/" + to_string(dimTask) + "D/"};

	ifstream scanV("files/" + to_string(dimTask) + "D/coefs.dat");
	scanV >> uk;
	scanV >> rk;
	scanV.close();

	int amntElements{0}, dimEps{0}, dimSigma{0};

	mesh.Construct(dimTask * amntNodes);
	for (double x : tmpBuf)
	{
		mesh.SetElement(tmpCount, x);
		tmpCount++;
	}

	tmpBuf.clear();
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
			for (int i = 0; i < 3; i++)
			{
				scan >> tmp;
				tmpBuf.push_back(tmp - 1);
			}
			amntElements++;
		}
		scan.close();
		elements.Construct(amntElements * 3);
		for (double x : tmpBuf)
		{
			elements.SetElement(tmpCount, x);
			tmpCount++;
		}
		tmpBuf.clear();
		tmpCount = 0;
		break;
	}
	default:
		printf("Wrong input: dimTask\n");
		break;
	}
	VectorSchwarz y(amntNodes * dimTask);
	MatrixSchwarz Eps(dimEps, amntElements);
	MatrixSchwarz Sigma(dimSigma, amntElements);

	strainMatrix S(dimTask, mesh, elements);

	MatrixSchwarz D(dimSigma, dimEps);
	D.Elastic_Modulus_Tensor(dimTask);

	CalcDisplacements(dimTask, &Route, y, mesh, elements, S, D, uk, amntSubdomains, stopCriteria, amntNodes, amntElements);
	Eps = S * y;
	/*for (int i = 0; i < y.GetSize() / dimTask; i++)
	{
		printf("%8.3g\t%8.3g\n",y[i * dimTask],y[i * dimTask + 1]);
	}*/

	Sigma = D * Eps;
	MatrixSchwarz SigmaT;
	Sigma.Transpose(SigmaT);
	Sigma.SetName("Sigma");

	Sigma.Record(Route, rk);
}