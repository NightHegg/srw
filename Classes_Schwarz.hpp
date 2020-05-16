#ifndef CLASSES_SCHWARZ_HPP
#define CLASSES_SCHWARZ_HPP

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "classes/Matrix.hpp"
#include "classes/Vector.hpp"
#include "classes/Basis_Functions.hpp"
#include "classes/Strain_Matrix.hpp"

using namespace std;

class VectorSchwarz : public Vector
{
private:
	std::vector<int> SchwarzNodes;
	int LeftBoundary, RightBoundary;
	bool UsingMethodSchwarz;
	std::string name;

public:
	VectorSchwarz() : Vector::Vector()
	{
		SchwarzNodes = std::vector<int>();
		LeftBoundary = RightBoundary = 0;
		UsingMethodSchwarz = false;
		name = "";
	}
	VectorSchwarz(int i) : Vector(i)
	{
		SchwarzNodes = std::vector<int>();
		LeftBoundary = 0;
		RightBoundary = iV - 1;
		UsingMethodSchwarz = false;
		name = "";
	}
	void Construct(int i)
	{
		Vector::Construct(i);
		LeftBoundary = 0;
		RightBoundary = iV - 1;
		UsingMethodSchwarz = false;
		name = "";
	}
	/*void Vector(const Vector &N)
	{
		Vector::Vector(N);
		LeftBoundary = 0;
		RightBoundary = iV - 1;
		UsingMethodSchwarz = false;
	}*/

	void Decomposition(int Amount_Subdomains, double *Coef_Overflow)
	{
		setlocale(LC_ALL, "Russian");
		int Count{0};
		int UserChoice{0};
		int Center_Subdomain{0};
		double TestValue1{0};
		double TestValue2{0};
		double CoefChosen{0};
		std::vector<double> CoefSuitable;
		std::vector<double> CoefVariants;
		for (int i = 1; i < 49; i++)
		{
			CoefVariants.push_back(0.01 * i);
		}
		double Length_Subdomain{V[iV / Amount_Subdomains] - V[0]};
		for (auto it : CoefVariants)
		{
			for (int i = 1; i < iV / Amount_Subdomains; i++)
			{
				if (fabs(V[i] - V[0] - it * Length_Subdomain) < 1e-15)
				{
					CoefSuitable.push_back(it);
				}
			}
		}
		if (CoefSuitable.size() > 1)
		{
			printf("There are few coefs for choice:\n");
			for (auto it : CoefSuitable)
			{
				printf("%d %.2f\n", ++Count, it);
			}
			printf("Choose one of coefs:\n");
			scanf_s("%d", &UserChoice);
			printf("\n");
			CoefChosen = CoefSuitable.at(UserChoice - 1);
			*Coef_Overflow = CoefChosen;
		}
		else if (CoefSuitable.size() == 1)
		{
			printf("There is only one coef for choice: %.2f\n", CoefSuitable.front());
			CoefChosen = CoefSuitable.front();
			*Coef_Overflow = CoefChosen;
		}
		else
		{
			printf("Coefs from 0.1 to 0.4 are inavailable!\n");
			system("PAUSE");
		}
		SchwarzNodes.push_back(0);
		for (int i = 1; i < Amount_Subdomains; i++)
		{
			Center_Subdomain = iV * i / Amount_Subdomains;
			TestValue1 = V[Center_Subdomain] - CoefChosen * Length_Subdomain;
			TestValue2 = V[Center_Subdomain] + CoefChosen * Length_Subdomain;
			for (int j = 0; j < iV; j++)
			{
				if (fabs(V[j] - TestValue1) < 1e-15 || fabs(V[j] - TestValue2) < 1e-15)
				{
					SchwarzNodes.push_back(j);
				}
			}
		}
		SchwarzNodes.push_back(iV - 1);
	}

	VectorSchwarz CreateAllocatedArray(int SchwarzStep)
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
			LeftB = SchwarzNodes[SchwarzStep * 2 - 1];
			RightB = SchwarzNodes[SchwarzStep * 2 + 2];
		}
		int AmountElements = RightB - LeftB + 1;
		VectorSchwarz A(AmountElements);
		for (int i = 0; i < A.iV; i++)
		{
			A.V[i] = V[i + LeftB];
		}
		A.LeftBoundary = LeftB;
		A.RightBoundary = RightB;
		return A;
	}

	void ReturnAllocatedArrayResults(VectorSchwarz &yChosen, int SchwarzStep)
	{
		for (int i = 0; i < yChosen.iV; i++)
		{
			V[i + yChosen.LeftBoundary] = yChosen.V[i];
		}
	}

	void Equal_SchwarzNodes(VectorSchwarz &v)
	{
		for (auto it : v.SchwarzNodes)
		{
			SchwarzNodes.push_back(it);
		}
		UsingMethodSchwarz = true;
	}

	bool Condition_Schwarz()
	{
		return (UsingMethodSchwarz == true ? true : false);
	}

	bool Compare_Boundary_Left(VectorSchwarz &v)
	{
		return (LeftBoundary == v.LeftBoundary ? true : false);
	}

	bool Compare_Boundary_Right(VectorSchwarz &v)
	{
		return (RightBoundary == v.RightBoundary ? true : false);
	}

	double ConvergenceL2(VectorSchwarz &yprev, VectorSchwarz &rr)
	{
		double s{0};
		double h = rr[1] - rr[0];
		double Length = rr[rr.iV - 1] - rr[0];
		double sum{0.0};
		for (int i = 0; i < iV; i++)
		{
			if (i == 0)
			{
				s = (rr.V[i + 1] - rr.V[i]) / 2;
			}
			else if (i == iV - 1)
			{
				s = (rr.V[i] - rr.V[i - 1]) / 2;
			}
			else
			{
				s = (((rr.V[i] - rr.V[i - 1]) / 2) + ((rr.V[i + 1] - rr.V[i]) / 2)) / 2;
			}
			sum += pow((V[i] - yprev.V[i]) / V[i], 2) * s * 1.0 / Length * 1.0;
		}
		return sqrt(sum);
	}

	void Partition(int dimTask)
	{
		double a, b;
		ifstream out("files/" + to_string(dimTask) + "D/nodes.dat");

		switch (dimTask)
		{
		case 1:
		{
			out >> a;
			out >> b;
			double h;
			V[0] = a;
			for (int i = 1; i < iV; i++)
			{
				h = (V[i] - V[i - 1]) / (iV - 1);
				V[i] = V[i - 1] + h;
			}
			break;
		}
		case 2:
		{

			break;
		}
		}
	}

	void SetName(std::string str)
	{
		name = str;
	}

	void Record(std::string Route, int amntSubdomains, double Coef)
	{
		
		std::string sep = "_";
		std::string size = std::to_string(iV - 1);
		std::string AS = std::to_string(amntSubdomains);

		if (iV < 10)
		{
			size = "00" + size;
		}
		else if (iV >= 10 && iV < 100)
		{
			size = "0" + size;
		}

		if (amntSubdomains < 2)
		{
			Route += name + sep + size + ".dat";
		}
		else
		{
			Route += name + sep + size + sep + AS + ".dat";
		}
		std::ofstream outfile(Route);
		for (int i = 0; i < iV; i++)
		{
			outfile << V[i] * Coef;
			outfile << std::endl;
		}
		outfile.close();
	}
};

class MatrixSchwarz : public Matrix
{
private:
	std::string name;

public:
	MatrixSchwarz() : Matrix::Matrix()
	{
		name = "";
	}
	MatrixSchwarz(int i, int j) : Matrix(i, j)
	{
		name = "";
	}
	void Construct(int i, int j)
	{
		Matrix::Construct(i, j);
		name = "";
	}

	void Elastic_Modulus_Tensor(int dimTask)
	{
		double E, nyu, lambda, myu;
		ifstream out("files/" + std::to_string(dimTask) + "D/material.dat");
		out >> E;
		out >> nyu;
		lambda = (nyu * E) / ((1 + nyu) * (1 - 2 * nyu) * 1.0);
		myu = E / (2 * (1 + nyu));

		switch (dimTask)
		{
		case 1:
			for (int i = 0; i < iM; i++)
			{
				for (int j = 0; j < jM; j++)
				{
					if (i == j)
						M[i][j] = lambda + 2 * myu;
					else
						M[i][j] = lambda;
				}
			}
			break;
		case 2:
			for (int i = 0; i < iM - 1; i++)
			{
				for (int j = 0; j < jM - 1; j++)
				{
					if (i == j)
						M[i][j] = lambda + 2 * myu;
					else
						M[i][j] = lambda;
				}
				M[iM][jM] = 2 * myu;
				break;
			}
		default:
			printf("Wrong input, matrix D\n");
			break;
		}
	}

	friend MatrixSchwarz operator*(MatrixSchwarz &N, MatrixSchwarz &L)
	{
		MatrixSchwarz P(N.iM, L.jM);
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

	void Create_Sy(strainMatrix &S, VectorSchwarz &m)
	{
		double h{0};
		Construct(S.iSize, m.GetSize());
		for (int j = 0; j < GetSize_j(); j++)
		{
			h = S.arg.GetElement(j + 1) - S.arg.GetElement(j);
			switch (S.dimTask)
			{
			case 1:
				M[0][j] = (m.GetElement(j + 1) - m.GetElement(j)) / h;
				M[1][j] = (m.GetElement(j + 1) + m.GetElement(j)) / (S.arg.GetElement(j + 1) + S.arg.GetElement(j));
				break;
			}
		}
	}

	void Create_B(strainMatrix &S, basfuncMatrix &matrN)
	{
		double h = matrN.arg.GetElement(matrN.numNode + 1) - matrN.arg.GetElement(matrN.numNode);
		Construct(S.iSize, S.dimTask * matrN.amntBE);
		for (int j = 0; j < GetSize_j(); j++)
		{
			switch (S.dimTask)
			{
			case 1:
				M[0][j] = (matrN.Get_N(matrN.node + h, j) - matrN.Get_N(matrN.node, j)) / h;
				M[1][j] = matrN.Get_N(matrN.node, j) / matrN.node;
				break;
			case 2:
				break;
			default:
				printf("Wrong input.\n");
				break;
			}
		}
	}

	friend MatrixSchwarz operator*(const MatrixSchwarz &N, const double val)
	{
		MatrixSchwarz P(N.iM, N.jM);
		for (int i = 0; i < N.iM; i++)
		{
			for (int j = 0; j < N.jM; j++)
			{
				P.M[i][j] = N.M[i][j] * val;
			}
		}
		return P;
	}

	friend MatrixSchwarz operator+(const MatrixSchwarz &N, const MatrixSchwarz &L)
	{
		MatrixSchwarz P(N.iM, N.jM);
		for (int i = 0; i < N.iM; i++)
		{
			for (int j = 0; j < N.jM; j++)
			{
				P[i][j] = N.M[i][j] + L.M[i][j];
			}
		}
		return P;
	}

	void SetName(std::string str)
	{
		name = str;
	}

	std::string GetName()
	{
		return name;
	}

	void Record(std::string Route, int amntNodes, int amntSubdomains, double Coef)
	{
		std::string sep = "_";
		std::string size = std::to_string(amntNodes-1);
		std::string AS = std::to_string(amntSubdomains);

		if (amntNodes < 10)
		{
			size = "00" + size;
		}
		else if (amntNodes >= 10 && amntNodes < 100)
		{
			size = "0" + size;
		}

		if (amntSubdomains < 2)
		{
			Route += name + sep + size + ".dat";
		}
		else
		{
			Route += name + sep + size + sep + AS + ".dat";
		}
		std::ofstream outfile(Route);
		for (int j = 0; j < jM; j++)
		{
			for (int i = 0; i < iM; i++)
			{
				outfile << M[i][j] * Coef;
				outfile << " ";
			}
			outfile << std::endl;
		}
		outfile.close();
	}
};

#endif