#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include "classes.h"

using namespace std;

void Get_Eps(Vector& rr, Vector& y, Matrix& Eps)
{
	double h = rr[1] - rr[0];
	for (int i = 0; i < Eps.jM; i++) //Цикл записи данных в файл
	{
		Eps[0][i] = (y[i + 1] - y[i])*1.0 / (h*1.0); //Радиальная деформация
		Eps[1][i] = (y[i + 1] + y[i])*1.0 / (rr[i] + rr[i + 1])*1.0; //Окружная деформация
																	 //Eps[i][1] = (y[i]) / (rr[i]); //Окружная деформация
	}
}

void Get_Sigma(Matrix& D, Matrix& Eps, Matrix& Sigma)
{
	Sigma = D * Eps;
	/*for (int i = 0; i < Eps.iM; i++) //Цикл записи данных в файл
	{

		Sigma[i][0] = K1*(Eps[i][0])*1.0 + K2*(Eps[i][1])*1.0;
		Sigma[i][1] = K2*(Eps[i][0] )*1.0 + K1*(Eps[i][1])*1.0;
		//Sigma[i][2] = (1.0 / 2.0)*(Sigma[i][0] + Sigma[i][1]);
	}*/
}