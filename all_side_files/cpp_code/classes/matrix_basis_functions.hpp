#ifndef BASIS_FUNCTIONS_HPP
#define BASIS_FUNCTIONS_HPP

#include "vector.hpp"

#include <vector>
#include <iostream>

using namespace std;

class basfuncMatrix
{
public:
	double node;
	int numNode;
	int dimTask;
	Vector func;
	Vector arg;
	std::vector<double> N;
	int amntBE;

	basfuncMatrix(int _dimTask, Vector a, int _numNode, double _node)
	{
		node = _node;
		numNode = _numNode;
		dimTask = _dimTask;
		arg = a;
		switch (dimTask)
		{
		case 1:
			amntBE = 2;
			break;
		case 2:
			amntBE = 3;
			break;
		}
	}
	double Get_N(double valNode, int val)
	{
		double h;
		double res;
		switch (dimTask)
		{
		case 1:
		{
			h = arg.GetElement(numNode + 1) - arg.GetElement(numNode);
			N.push_back((arg.GetElement(numNode + 1) - valNode) / h);
			N.push_back((valNode - arg.GetElement(numNode)) / h);
			res = N[val];
			N.erase(N.begin(), N.end());
			break;
		}
		}
		return res;
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