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