#pragma once

#include "string"
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

void Record_AddData(int *N, int* Amount_Subdomains, int *Counter, double *stopCriteria)
{
    ofstream ofile("files/addData.dat");
    ofile<<*N;
    ofile<<endl;
    ofile<<*Amount_Subdomains;
    ofile<<endl;
    ofile<<*Counter;
    ofile<<endl;
    ofile<<*stopCriteria;
    ofile<<endl;
}
