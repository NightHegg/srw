#include <iostream>
#include <string>
#include "classes/Basis_Functions.hpp"
#include "Classes_Schwarz.hpp"
#include "Num_Integration.hpp"

/*void formElemMatStiffness(int i, 
                            MatrixSchwarz& KM, 
                            VectorSchwarz& rVec, 
                            int amntNodes,
                            MatrixSchwarz& D,
                            std::string typeIntegration)
{
    MatrixSchwarz localKM;
    Basis_Functions ElementB(i, rVec, amntNodes);
    Numerical_Integration(i, rVec, D, ElementB, typeIntegration, localKM);
    for (int j = 0; j < iB; j++)
    {
        for (int k = 0; k < jB; k++)
        {
            KM[i + j][i + k] += localKM[j][k];
        }
    }
}*/

void Form_Element_Mat_Stiffness(int dimTask,
                                VectorSchwarz a,
                                VectorSchwarz y,
                                VectorSchwarz D,
                                VectorSchwarz S)
{
    for (int i = 0; i < SizeDomain - 1; i++)
    {
        Basis_Functions ElementB(i, rrChosen, TaskAmNodes);
        Numerical_Integration(i, rrChosen, D, ElementB, Type_Integration, A);
    }
}

void Ensembling()
{
}

void Get_Displacements(int dimTask, VectorSchwarz y)
{
    double tmp;
    int amntElements{0};
    VectorSchwarz a;

    MatrixSchwarz K(amntElements,amntElements);
    VectorSchwarz F(amntElements);

    for (int i = 0; i < amntElements; i++)
    {
        Form_Elem_Mat_Stiffness(Ke, i);
        Form_Glob_Mat_Stiffness(K, Ke, i);
        Form_Elem_Vec_Right(Fe, i);
        Form_Glob_Vec_Right(F, Fe, i);
        Form_Boundary_Conditions(K, F);
    }

    Ensembling(K, F);
    Tridiogonal_Algorithm(SizeDomain, K, F, y);
}