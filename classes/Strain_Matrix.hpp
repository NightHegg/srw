#ifndef STRAIN_MATRIX_HPP
#define STRAIN_MATRIX_HPP

#include "classes/Vector.hpp"
#include <vector>
/**
 * ! Переписать S, чтобы сначала была просто пустышка, а потом сверху добавлялась функция в зависимости от размерности
 **/

class MatrixStrain
{
public:
	int dimSol;
	int numNode;
	int dimTask;
	Vector func;
	Vector arg;
	double **S;

	MatrixStrain(int _dimTask, Vector &a)
	{
		numNode = 0;
		dimTask = _dimTask;
		arg = a;
		switch (dimTask)
		{
		case 1:
		{
			dimSol=2;
			S = new double *[2];
			for (int i = 0; i < 2; i++)
				S[i] = new double[1];

			S[0][0] = Derivative_BE(func, arg, numNode);
			S[1][0] = func.GetElement(numNode) / (arg.GetElement(numNode));
			break;
		}
		}
	}
	double Derivative_BE(Vector &func, Vector &arg, int i)
	{
		return (func.GetElement(i + 1) - func.GetElement(i)) / (arg.GetElement(i + 1) - arg.GetElement(i));
	}
};

#endif