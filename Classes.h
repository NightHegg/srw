#pragma once
#include "pch.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>

#ifndef _INTERACTIONS_MATRIX_H
#define _INTERACTIONS_MATRIX_H

using namespace std;

class Vector
{
public:
	double* V;
	int iV;
	vector<int> SchwarzNodes;
	int LeftBoundary;
	int RightBoundary;
	bool UsingMethodSchwarz;
	Vector()
	{
		this->iV = 0;	
		this->V = nullptr;
		SchwarzNodes =vector<int> ();
		LeftBoundary = RightBoundary = 0;
		UsingMethodSchwarz = false;
	}
	Vector(int i)
	{
		iV = i;
		V = new double[iV];

		NullVector();

		SchwarzNodes = vector<int> ();
		LeftBoundary = 0;
		RightBoundary = iV - 1;
		UsingMethodSchwarz = false;
	}
	void NullVector()
	{
		for (int i = 0; i < iV; i++)
		{
			this->V[i] = 0;
		}
	}
	~Vector()
	{
		delete[] V;
	}
	void Construct(int i)
	{
		this->iV = i;
		V = new double[iV];
		LeftBoundary = 0;
		RightBoundary = iV-1;
		UsingMethodSchwarz = false;
	}
	Vector(const Vector& N)
	{
		Construct(N.iV);
		for (int i = 0; i < iV; i++)
		{
			V[i] = N.V[i];
		}
		LeftBoundary = N.LeftBoundary;
		RightBoundary = N.RightBoundary;
		UsingMethodSchwarz = 0;
	}
	void operator= (Vector N)
	{
		Construct(N.iV);
		for (int i = 0; i < iV; i++)
		{
			V[i] = N.V[i];
		}
		LeftBoundary = N.LeftBoundary;
		RightBoundary = N.RightBoundary;
	}
	void Show()
	{
		for (int i = 0; i < iV; i++)
		{
			printf("%g %d\n", V[i],i);
		}
	}
	double &operator [] (int _i)
	{
		return V[_i];
	}
	double operator-(double i)
	{
		return V[iV] - i;
	}
	double operator-(Vector A)
	{
		return V[iV] - A[A.iV];
	}
	void Decomposition(int Amount_Subdomains)
	{
		setlocale(LC_ALL, "Russian");
		int UserChoice{ 0 };
		int Center_Subdomain{ 0 };
		double TestValue1{ 0 };
		double TestValue2{ 0 };
		double CoefChosen{ 0 };
		vector<double>CoefSuitable;
		vector<double> CoefVariants;
		for (int i = 1;i < 5;i++)
		{
			CoefVariants.push_back(0.1*i);
		}
		double Length_Subdomain{ V[iV / Amount_Subdomains] - V[0] };
		for (vector<double>::iterator it=CoefVariants.begin();it!=CoefVariants.end();++it)
		{
			for (int i = 1; i < iV / Amount_Subdomains; i++)
			{
				if (fabs(V[i]- V[0]-*it * Length_Subdomain)<1e-15)
				{
					CoefSuitable.push_back(*it);
				}
			}
		}
		if (CoefSuitable.size() > 1)
		{
			printf("�� ����� ������� ��������� ������������� ��� �������������� �������:\n");
			for (vector<double>::iterator it = CoefSuitable.begin();it != CoefSuitable.end();++it)
			{
				printf("%.1f\n", *it);
			}
			printf("�������� ���� �� �������������:\n");
			scanf_s("%d", &UserChoice);
			printf("\n");
			CoefChosen = CoefSuitable.at(UserChoice - 1);
		}
		else if(CoefSuitable.size()==1)
		{
			CoefChosen = CoefSuitable.front();
		}
		else
		{
			printf("������������ �� 0.1 �� 0.4 �� ��������!\n");
			system("PAUSE");
		}
		SchwarzNodes.push_back(0);
		for (int i = 1; i < Amount_Subdomains; i++)
		{
			Center_Subdomain = iV * i / Amount_Subdomains;
			TestValue1 = V[Center_Subdomain] - CoefChosen * Length_Subdomain;
			TestValue2 = V[Center_Subdomain] + CoefChosen * Length_Subdomain;
			for (int j = 0;j < iV;j++)
			{
				if (fabs(V[j]-TestValue1)<1e-15 || fabs(V[j] - TestValue2) < 1e-15)
				{
					SchwarzNodes.push_back(j);
				}
			}
		}
		SchwarzNodes.push_back(iV - 1);
	}

	Vector CreateAllocatedArray(int SchwarzStep)
	{
		int LeftB, RightB;
		if (SchwarzStep == 0)
		{
			LeftB = SchwarzNodes[SchwarzStep];
			RightB = SchwarzNodes[SchwarzStep + 2];
		}
		else if (SchwarzStep == (SchwarzNodes.size() / 2) - 1)
		{
			LeftB = SchwarzNodes[SchwarzStep * 2 - 1];
			RightB = SchwarzNodes[SchwarzStep * 2 + 1];
		}
		else
		{
			//printf("NUMBERS: %d\n", SchwarzNodes[SchwarzStep * 2 - 1]);
			LeftB = SchwarzNodes[SchwarzStep * 2 - 1];
			RightB = SchwarzNodes[SchwarzStep * 2 + 2];
		}
		int AmountElements = RightB - LeftB + 1;
		Vector A(AmountElements);
		for (int i = 0; i < A.iV; i++)
		{
			A.V[i] = V[i + LeftB];	
		}
		A.LeftBoundary = LeftB;
		A.RightBoundary = RightB;
		return A;
	}
	void ReturnAllocatedArrayResults(Vector& yChosen, int SchwarzStep)
	{
		for (int i = 0;i < yChosen.iV;i++)
		{
			V[i + yChosen.LeftBoundary] = yChosen.V[i];
		}
	}
	void FillVector(double var)
	{
		for (int i = 0;i < iV; i++)
		{
			this->V[i] = var;
		}
	}
	double ConvergenceM(Vector &A)
	{
		double max=0.0;
		for (int i = 0;i < iV;i++)
		{
			if ((abs(this->V[i]-A.V[i]))>max)
			{
				max = (abs(this->V[i] - A.V[i]));
			}
		}
		return max;
	}
	double ConvergenceL2(Vector &A, Vector &rr)
	{
		double h = rr[1] - rr[0];
		double Length = rr[rr.iV - 1] - rr[0];
		double sum{ 0.0 };
		for (int i = 0;i < iV;i++)
		{
			sum += pow((V[i] - A.V[i])/V[i], 2)*h*1.0/Length*1.0;
		}
		return sqrt(sum);
	}
};

class BasicElements
{
public:
	//Matrix M;
	double h;
	double* N;
	int iBE;
	Vector r;
	double Node;
	int AmNodes;
	//BasicElements(int i, Vector& rr, int AmountNodes, int EpsDimArray, int DimTask)
	BasicElements(int i, Vector& rr, int AmountNodes)
	{
		Node = NULL;
		this->r.Construct(rr.iV);
		for (int i = 0; i < rr.iV; i++)
		{
			this->r.V[i] = rr.V[i];
		}
		h = r[1] - r[0];
		AmNodes = AmountNodes;
		iBE = i;
		N = new double[AmNodes];
	}
	double Get_N(int i, double _Node)
	{
		N[0] = (this->r.V[iBE + 1] - _Node) / h;
		N[1] = (_Node - this->r.V[iBE]) / h;
		return N[i];
	}
	double Derivative_BE(int i,double _Node)
	{
		double Value1, Value2;
		Value1 = Get_N(i, _Node);
		Value2 = Get_N(i, _Node+h);
		return (Value2-Value1) / this->h;
	}
};

class Matrix
{
public:
	double** M;
	int iM, jM;
	Matrix()
	{
		iM = 0;
		jM = 0;
		M = NULL;
	}
	Matrix(int i, int j)
	{
		ConstructMatrix(i, j);
	}
	void ConstructMatrix(int i, int j)
	{
		this->iM = i;
		this->jM = j;
		M = new double*[iM];
		for (int i = 0; i < iM; i++)
			M[i] = new double[jM];

		NullMatrix();
	}					
	void NullMatrix()
	{
		for (int i = 0; i < iM; i++)
		{
			for (int j = 0; j < jM; j++)
			{
				this->M[i][j] = 0;
			}
		}
	}

	Matrix(const Matrix& N)
	{
		ConstructMatrix(N.iM, N.jM);
		for (int i = 0; i < iM; i++)
		{
			for (int j = 0; j < jM; j++)
			{
				this->M[i][j] = N.M[i][j];
			}
		}
	}
	~Matrix()
	{
		for (int i = 0;i < iM;i++)
			delete[] M[i];
		delete[] M;
	}
	void Show()
	{
		cout << endl;
		for (int i = 0; i < iM; i++)
		{
			for (int j = 0; j < jM; j++)
			{
				cout.width(12);
				cout << M[i][j] << "\t";
			}
			cout << "\n";
		}
		cout << endl;
	}
	void Transpose(Matrix& A)
	{
		A.ConstructMatrix(jM, iM);
		for (int i = 0; i < A.iM; i++)
		{
			for (int j = 0; j < A.jM; j++)
			{
				A.M[i][j] = this->M[j][i];
			}
		}
	}
	friend Matrix operator * (const Matrix& N,const Matrix& L)
	{
		Matrix P(N.iM, L.jM);
		for (int i = 0; i < N.iM; i++)
		{
			for (int j = 0; j < L.jM; j++)
			{
				P.M[i][j] = 0;
				for (int k = 0; k < N.jM; k++)
				{
					P.M[i][j] += (N.M[i][k] * L.M[k][j]);
				}
			}
		}
		return P;
	}
	friend Matrix operator * (const Matrix& N,const double val)
	{
		Matrix P(N.iM, N.jM);
		for (int i = 0; i < N.iM; i++)
		{
			for (int j = 0; j < N.jM; j++)
			{
				P.M[i][j] = N.M[i][j] * val;
			}
		}
		return P;
	}
	Matrix& operator = (const Matrix& N)
	{
		ConstructMatrix(N.iM, N.jM);
		for (int i = 0; i < iM; i++)
		{
			for (int j = 0; j < jM; j++)
			{
				this->M[i][j] = N.M[i][j];
			}
		}
		return *this;
	}

	double* operator [] (int i)
	{
		return M[i];
	}
	void ConstructFullB(BasicElements& BE,double _Node)
	{
		ConstructMatrix(2,2);
		/*M[0][0] = -1 / BE.h;
		M[0][1]= 1 / BE.h;
		M[1][0] = (BE.r[BE.iBE + 1] - _Node) / (BE.h*_Node);
		M[1][1] = (_Node- BE.r[BE.iBE]) / (BE.h*_Node);*/
		for (int j = 0; j < jM; j++)
		{
			M[0][j] = BE.Derivative_BE(j,_Node);
			M[1][j] = BE.Get_N(j,_Node) / _Node;
		}

	}
};

#endif _INTERACTIONS_MATRIX_H