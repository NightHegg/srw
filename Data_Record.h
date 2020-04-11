#pragma once

#include "string"
#include <fstream>
#include "Classes_Schwarz.hpp"

using namespace std;
/**
 * TODO: Recreate this function for every class //DONE
 */

void Record_AddData(int N, int amntSubdomains, int Counter, double stopCriteria, double Coef_Overflow, string Route)
{
    std::string sep = "_";
    std::string size = std::to_string(N);
    std::string name = "AddData";
    std::string AS = std::to_string(amntSubdomains);

    if (N < 10)
    {
        size = "00" + size;
    }
    else if (N >= 10 && N < 100)
    {
        size = "0" + size;
    }
    if (amntSubdomains < 2)
    {
        Route += name + sep + size + ".dat";
    }
    else
    {
        Route += name + sep + size + sep + AS + ".dat";
    }
    ofstream ofile(Route);
    ofile << N;
    ofile << endl;
    ofile << amntSubdomains;
    ofile << endl;
    ofile << Counter;
    ofile << endl;
    ofile << stopCriteria;
    ofile << endl;
    ofile << Coef_Overflow;
    ofile << endl;
}
