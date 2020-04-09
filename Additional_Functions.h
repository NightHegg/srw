
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include "Classes_Schwarz.hpp"

using namespace std;

void Get_Eps(VectorSchwarz& rr, VectorSchwarz& y, MatrixSchwarz& Eps)
{
	double h = rr[1] - rr[0];
	for (int i = 0; i < Eps.GetSize_j(); i++)
	{
		Eps[0][i] = (y[i + 1] - y[i])*1.0 / (h*1.0);
		Eps[1][i] = (y[i + 1] + y[i])*1.0 / (rr[i] + rr[i + 1])*1.0;
																	 //Eps[i][1] = (y[i]) / (rr[i]);
	}
}