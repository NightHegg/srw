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

	MatrixStrain(int _dimTask)
	{
		numNode = 0;
		dimTask = _dimTask;
	}
};

#endif