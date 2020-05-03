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
	double *N;
	double **B;
	Basis_Functions(int _dimTask, Vector a)
	{
		numNode = 0;
		dimTask = _dimTask;
		arg = a;
		switch (dimTask)
		{
		case 1:
		{
			B = new double *[2];
			for (int i = 0; i < 2; i++)
				B[i] = new double[3];

			N = new double[2];

			B[0][0] = Derivative_BE(func, arg, numNode);
			B[1][0] = func.GetElement(numNode) / (arg.GetElement(numNode));
			
			N[0] = (r.GetElement(iBE + 1) - _Node) / h;
			N[1] = (_Node - r.GetElement(iBE)) / h;
			break;
		}
		}
	}
	double Derivative_BE(Vector &func, Vector &arg, int i)
	{
		return (func.GetElement(i + 1) - func.GetElement(i)) / (arg.GetElement(i + 1) - arg.GetElement(i));
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