#ifndef ADDFUNCS_BOUNDARYCONDS_HPP
#define ADDFUNCS_BOUNDARYCONDS_HPP

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include "Num_Integration.hpp"
#include "Methods.hpp"
#include "classes.hpp"
#include "record_data.hpp"

void Form_Boundary_Conditions(int dimTask,
                              vector<double> &arrBound,
                              VectorSchwarz &y,
                              VectorSchwarz &mesh,
                              VectorSchwarz &elements,
                              int amntElements,
                              int amntNodes,
                              MatrixSchwarz &K,
                              VectorSchwarz &F)
{
    bool keyDirichlet{false};
    bool keyNeumann{false};
    int Coef{0};
    int altCoef{0};
    double sizeFace{0};
    double tmp{0};
    int size = mesh.GetSize();
    vector<double> coordsBound;
    vector<double> localMesh;
    vector<int> localElement;
    double A{0};
    vector<int> localFace;

    switch (dimTask)
    {
    case 1:
        F[0] += arrBound[2] * mesh[0];
        F[size - 1] += (-1.0) * arrBound[3] * mesh[size - 1];
        if (arrBound[0] != -1)
        {
            F[1] = F[1] - K[1][0] * arrBound[0];
            K[0][0] = 1;
            K[0][1] = 0;
            K[1][0] = 0;
            F[0] = arrBound[0];
        }
        if (arrBound[1] != -1)
        {
            F[size - 2] = F[size - 2] - K[size - 2][size - 1] * arrBound[size - 1];
            K[size - 1][size - 1] = 1;
            K[size - 1][size - 2] = 0;
            K[size - 2][size - 1] = 0;
            F[size - 1] = arrBound[size - 1];
        }
        break;
    case 2:
        ifstream scan("files/" + to_string(dimTask) + "D/nodes.dat");
        while (!scan.eof())
        {
            scan >> tmp;
            coordsBound.push_back(tmp);
        }
        for (int i = 0; i < 4; i++)
        {
            if ((i == 0 || i == 2) && (arrBound[i] != -1))
            {
                Coef = 1;
                altCoef = 0;
                keyDirichlet = true;
            }
            if ((i == 0 || i == 2) && (arrBound[i + 4] != -1))
            {
                Coef = 1;
                altCoef = 0;
                keyNeumann = true;
            }
            if ((i == 1 || i == 3) && (arrBound[i] != -1))
            {
                Coef = 0;
                altCoef = 1;
                keyDirichlet = true;
            }
            if ((i == 1 || i == 3) && (arrBound[i + 4] != -1))
            {
                Coef = 0;
                altCoef = 1;
                keyNeumann = true;
            }
            if (keyDirichlet)
            {
                for (int j = 0; j < mesh.GetSize() / 2; j++)
                {
                    if (mesh.GetElement(j * dimTask + Coef) == coordsBound[i * dimTask + Coef])
                    {
                        for (int k = 0; k < mesh.GetSize(); k++)
                        {
                            K[j * dimTask + Coef][k] = 0;
                            K[k][j * dimTask + Coef] = 0;
                        }
                        K[j * dimTask + Coef][j * dimTask + Coef] = 1;
                        F[j * dimTask + Coef] = arrBound[i];
                        /* for (int k = 0; k < mesh.GetSize(); k++)
                        {
                            if (k != j * dimTask + Coef)
							{
								F[k] = F[k] - K[k][j * dimTask + Coef] * arrBound[i];
								K[k][j * dimTask + Coef] = 0;
							}
                        }*/
                    }
                }
                keyDirichlet = false;
            }
            // !ДОБАВИТЬ УСЛОВИЕ ПРОВЕРКИ ДЛЯ LOCALFACE
            if (keyNeumann)
            {
                for (int j = 0; j < amntElements; j++)
                {
                    FormLocalElement(elements, localElement, j);
                    FormLocalMesh(localElement, mesh, localMesh);
                    for (int k = 0; k < localMesh.size() / dimTask; k++)
                    {
                        if (localMesh[k * dimTask + Coef] == coordsBound[i * dimTask + Coef])
                            localFace.push_back(k);
                        if (localFace.size() == 2)
                        {
                            sizeFace = abs(localMesh[localFace[0] * dimTask + altCoef] - localMesh[localFace[1] * dimTask + altCoef]);
                            for (auto x : localFace)
                            {
                                F[localElement[x] * dimTask + Coef] = F[localElement[x] * dimTask + Coef] + sizeFace * arrBound[i + 4] / 2;
                            }
                        }
                    }
                    localFace.clear();
                    localElement.clear();
                    localMesh.clear();
                }
                keyNeumann = false;
            }
        }
        break;
    }
}

void Form_Boundary_Conditions_Schwarz(int dimTask,
                                      vector<double> &arrBound,
                                      VectorSchwarz &y,
                                      VectorSchwarz &ySubd,
                                      VectorSchwarz &yPreviousSubd,
                                      VectorSchwarz &mesh, MatrixSchwarz &K,
                                      VectorSchwarz &F)
{
    int size = ySubd.GetSize();

    switch (dimTask)
    {
    case 1:
        if (ySubd.Compare_Boundary_Left(y))
        {
            F[size - 2] = F[size - 2] - K[size - 2][size - 1] * yPreviousSubd[size - 1];
            K[size - 1][size - 1] = 1;
            K[size - 1][size - 2] = 0;
            K[size - 2][size - 1] = 0;
            F[size - 1] = yPreviousSubd[size - 1] * 1.0;

            F[0] += arrBound[2] * mesh[0];
        }
        else if (ySubd.Compare_Boundary_Right(y))
        {
            F[1] = F[1] - K[1][0] * ySubd[0];
            K[0][0] = 1;
            K[0][1] = 0;
            K[1][0] = 0;
            F[0] = ySubd[0];

            F[size - 1] += (-1.0) * arrBound[3] * mesh[size - 1];
        }
        else
        {
            F[1] = F[1] - K[1][0] * ySubd[0];
            K[0][0] = 1;
            K[0][1] = 0;
            K[1][0] = 0;
            F[0] = ySubd[0];

            F[size - 2] = F[size - 2] - K[size - 2][size - 1] * yPreviousSubd[size - 1];
            K[size - 1][size - 1] = 1;
            K[size - 1][size - 2] = 0;
            K[size - 2][size - 1] = 0;
            F[size - 1] = yPreviousSubd[size - 1] * 1.0;
        }
        break;
    }
}

#endif