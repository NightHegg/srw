#ifndef BASIS_FUNCTIONS_HPP
#define BASIS_FUNCTIONS_HPP

#include "classes/Vector.hpp"

class Basis_Functions
{
public:
	double h;
	double* N;
	int iBE;
	Vector r;
	double Node;
	int AmNodes;
	Basis_Functions(int i, Vector& rr, int AmountNodes)
	{
		Node = NULL;
		r=rr;
		h = r[1] - r[0];
		AmNodes = AmountNodes;
		iBE = i;
		N = new double[AmNodes];
	}
	double Get_N(int i, double _Node)
	{
		N[0] = (r.GetElement(iBE+1) - _Node) / h;
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