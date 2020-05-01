#ifndef BASIS_FUNCTIONS_HPP
#define BASIS_FUNCTIONS_HPP

#include "classes/Vector.hpp"
#include <vector>

class Basis_Functions
{
public:
	int numNode;
	int dimTask;
	Vector func;
	Vector arg;
	std::vector<double> N;
	vector<vector<double>> B;
	Basis_Functions(int dT)
	{
		numNode=0;
		dimTask = dT;
		Vector func;
		Vector arg;
		switch (dimTask)
		{
		case 1:
		{
			vector<double> tmp;
			tmp.push_back(Derivative_BE(func, arg, numNode));
			B.push_back(tmp);
			vector<double> tmp2;
			tmp2.push_back(func.GetElement(numNode)/(arg.GetElement(numNode)));
			B.push_back(tmp2);
			break;
		}
		}
	}
	double Derivative_BE(Vector &func, Vector &arg, int i)
	{
		return (func.GetElement(i + 1) - func.GetElement(i)) / (arg.GetElement(i+1)-arg.GetElement(i));
	}
};

class Basis_Functions
{
public:
	double h;
	double *N;
	int iBE;
	Vector r;
	double Node;
	int AmNodes;
	Basis_Functions(int i, Vector &rr, int AmountNodes)
	{
		Node = NULL;
		r = rr;
		h = r[1] - r[0];
		AmNodes = AmountNodes;
		iBE = i;
		N = new double[AmNodes];
	}
	double Get_N(int i, double _Node)
	{
		N[0] = (r.GetElement(iBE + 1) - _Node) / h;
		N[1] = (_Node - r.GetElement(iBE)) / h;
		return N[i];
	}
	double Derivative_BE(int i, double _Node)
	{
		double Value1, Value2;
		Value1 = Get_N(i, _Node);
		Value2 = Get_N(i, _Node + h);
		return (Value2 - Value1) / this->h;
	}
};

#endif