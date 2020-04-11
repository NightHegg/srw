#include <iostream>
#include "classes/Basis_Functions.hpp"

/*FormElementMatrixStiffness(int i, Vector r)
{
    Basis_Functions ElementB(i, rrChosen, TaskAmNodes);
    Numerical_Integration(i, rrChosen, D, ElementB, Type_Integration, A);
    for (int j = 0; j < iB; j++)
    {
        for (int k = 0; k < jB; k++)
        {
            KM[i + j][i + k] += A[j][k];
        }
    }
}*/