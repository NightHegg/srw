#include <iostream>
#include <string>
#include "classes/Basis_Functions.hpp"
#include "Classes_Schwarz.hpp"
#include "Num_Integration.hpp"
#include "Methods.hpp"

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

void Form_Elem_Mat_Stiffness(int dimTask,
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

void Ensembling(MatrixSchwarz K, VectorSchwarz F, MatrixSchwarz D, int amntElements)
{
    MatrixSchwarz Ke(D.GetSize_i(), D.GetSize_i());
    VectorSchwarz Fe(D.GetSize_i());

    for (int i = 0; i < amntElements; i++)
    {
        Form_Elem_Mat_Stiffness(Ke, i);
        Form_Glob_Mat_Stiffness(K, Ke, i);
        Form_Elem_Vec_Right(Fe, i);
        Form_Glob_Vec_Right(F, Fe, i);
        Form_Boundary_Conditions(K, F);
    }
}

void Get_Displacements(int dimTask, VectorSchwarz y, VectorSchwarz a, MatrixSchwarz B, MatrixSchwarz D)
{
    int amntNodes = y.GetSize();
    int amntElements;
    switch (dimTask)
    {
    case 1:
    {
        amntElements = amntNodes - 1;
    }
    case 2:
    {
        ifstream scan("files/2D/elements.dat");
        while (!scan.eof())
        {
            amntElements++;
        }
        scan.close();
    }
    }

    MatrixSchwarz K(amntNodes, amntNodes);
    VectorSchwarz F(amntNodes);

    Ensembling(K, F, D, amntElements);
    Tridiogonal_Algorithm(amntNodes, K, F, y);
}