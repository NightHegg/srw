#ifndef BASIS_FUNCTIONS_HPP
#define BASIS_FUNCTIONS_HPP

#include "classes/Vector.hpp"
#include <vector>

class Basis_Functions
{
public:
	int numNode;
	double Node;
	int dimTask;
	Vector func;
	Vector arg;
	vector<double> N;

	Basis_Functions(int _dimTask, Vector a)
	{
		numNode = 0;
		Node = 0;
		dimTask = _dimTask;
		arg = a;
		switch (dimTask)
		{
		case 1:
		{
			N.push_back((a.GetElement(numNode + 1) - Node) / (a.GetElement(numNode + 1) - a.GetElement(numNode)));
			N.push_back((Node - a.GetElement(numNode + 1)) / (a.GetElement(numNode + 1) - a.GetElement(numNode)));
			break;
		}
		}
	}
};

/*class Basis_Functions
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
};*/

#endif