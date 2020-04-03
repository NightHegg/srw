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
    if (Sigma.GetSize_j()<10)
    {
        size="00"+size;
    }
    else if (Sigma.GetSize_j()>=10 && Sigma.GetSize_j()<100)
    {
        size="0"+size;
    }
    std::string AS=std::to_string(Amount_Subdomains);
    ofstream outfile(Route+"y"+sep+size+sep+AS+".dat");
    for (int i=0;i<y.GetSize();i++)
    {
        outfile <<y.GetElement(i) * uk;
        outfile <<endl;
    }
    outfile.close();
    ofstream outfile_second(Route+"Sigma"+sep+size+sep+AS+".dat");
    for (int j=0;j<Sigma.GetSize_j();j++)
    {
        outfile_second<<Sigma.GetElement(0,j) * rk<<" "<<Sigma.GetElement(1,j) * rk;
        outfile_second<<endl;
    }
    outfile_second.close();
}

void Record_AddData(int N, int Amount_Subdomains, int Counter, double stopCriteria, double Coef_Overflow, string Route)
{
    std::string sep="_";
    std::string size=std::to_string(N);
    if (N<10)
    {
        size="00"+size;
    }
    else if (N>=10 && N<100)
    {
        size="0"+size;
    }
    std::string AS=std::to_string(Amount_Subdomains);
    ofstream ofile(Route+"AddData"+sep+size+sep+AS+".dat");
    ofile<<Amount_Subdomains;
    ofile<<endl;
    ofile<<Counter;
    ofile<<endl;
    ofile<<stopCriteria;
    ofile<<endl;
    ofile<<Coef_Overflow;
    ofile<<endl;
}
