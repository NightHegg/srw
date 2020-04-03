#pragma once

#include "string"
#include <fstream>
#include "Classes_Schwarz.hpp"

using namespace std;
/**
 * TODO: Recreate this function for every class
 */
void Record_Results(Vector& y, Matrix& Sigma, double uk, double rk, int Amount_Subdomains, string Route)
{
    std::string sep="_";
    std::string size=std::to_string(Sigma.GetSize_j());
    std::string AS=std::to_string(Amount_Subdomains);
    ofstream outfile(Route+size+sep+AS+sep+"y"+".dat");
    for (int i=0;i<y.GetSize();i++)
    {
        outfile <<y.GetElement(i) * uk;
        outfile <<endl;
    }
    outfile.close();
    ofstream outfile_second(Route+size+sep+AS+sep+"Sigma"+".dat");
    for (int j=0;j<Sigma.GetSize_j();j++)
    {
        outfile_second<<Sigma.GetElement(0,j) * rk<<" "<<Sigma.GetElement(1,j) * rk;
        outfile_second<<endl;
    }
    outfile_second.close();
}

void Record_AddData(int N, int Amount_Subdomains, int Counter, double stopCriteria, double Coef_Overflow, string Route)
{
    std::string size=std::to_string(N);
    std::string AS=std::to_string(Amount_Subdomains);
    ofstream ofile(Route+size+"_"+AS+"_addData.dat");
    ofile<<Amount_Subdomains;
    ofile<<endl;
    ofile<<Counter;
    ofile<<endl;
    ofile<<stopCriteria;
    ofile<<endl;
    ofile<<Coef_Overflow;
    ofile<<endl;
}
