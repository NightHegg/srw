#ifndef STRAIN_MATRIX_HPP
#define STRAIN_MATRIX_HPP

#include "classes/Vector.hpp"
/**
 * ! Переписать S, чтобы сначала была просто пустышка, а потом сверху добавлялась функция в зависимости от размерности
 **/

class strainMatrix
{
public:
	int dimTask;
	int iSize;
	Vector arg;
	strainMatrix(int _dimTask, Vector &a)
	{
		arg = a;
		dimTask = _dimTask;
		switch (dimTask)
		{
		case 1:
			iSize = 2;
			break;
		case 2:
			iSize = 3;
			break;
		case 3:
			iSize = 4;
			break;
		default:
			printf("Wrong input.\n");
			break;
		}
	}
};

#endif