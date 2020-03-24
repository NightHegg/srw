#pragma once

#include <fstream>
#include "Classes.h"
/**
 * TODO: Recreate this function for every class
 */
void Record_Results(Vector& y, Matrix& Sigma, double uk, double rk)
{
    ofstream outfile("results/y.dat");
    for (int i=0;i<y.iV;i++)
    {
        outfile <<y.V[i] * uk;
        outfile <<endl;
    }
    outfile.close();
    ofstream outfile_second("results/Sigma.dat");
    for (int j=0;j<Sigma.jM;j++)
    {
        outfile_second<<Sigma.M[0][j] * rk<<" "<<Sigma.M[1][j] * rk;
        outfile_second<<endl;
    }
    outfile_second.close();
}
