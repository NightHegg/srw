#ifndef MATRIX_SCHWARZ_HPP
#define MATRIX_SCHWARZ_HPP

#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#include "matrix.hpp"
#include "vector_schwarz.hpp"
#include "matrix_strain.hpp"
#include "matrix_basis_functions.hpp"

class MatrixSchwarz : public Matrix
{
private:
    std::string name;

public:
    MatrixSchwarz();
    MatrixSchwarz(int i, int j);
    void Construct(int i, int j);

    void Elastic_Modulus_Tensor(int dimTask);

    friend MatrixSchwarz operator*(MatrixSchwarz &N, MatrixSchwarz &L);
    friend MatrixSchwarz operator*(const MatrixSchwarz &N, const double val);
    friend MatrixSchwarz operator+(const MatrixSchwarz &N, const MatrixSchwarz &L);
    friend VectorSchwarz operator*(MatrixSchwarz &A, VectorSchwarz &b);

    void Create_Sy(strainMatrix &S, VectorSchwarz &m);
    void Create_B(strainMatrix &S, basfuncMatrix &matrN);

    void SetName(std::string str);
    std::string GetName();

    void Record(std::string Route, int amntNodes, int amntSubdomains, double Coef);
};

MatrixSchwarz::MatrixSchwarz() : Matrix()
{
    name = "";
}

MatrixSchwarz::MatrixSchwarz(int i, int j) : Matrix(i, j)
{
    name = "";
}

void MatrixSchwarz::Construct(int i, int j)
{
    Matrix::Construct(i, j);
    name = "";
}

void MatrixSchwarz::Elastic_Modulus_Tensor(int dimTask)
{
    double E, nyu, lambda, myu;
    ifstream out("files/" + std::to_string(dimTask) + "D/material.dat");
    out >> E;
    out >> nyu;
    lambda = (nyu * E) / ((1 + nyu) * (1 - 2 * nyu) * 1.0);
    myu = E / (2 * (1 + nyu));
    out.close();

    switch (dimTask)
    {
    case 1:
        for (int i = 0; i < iM; i++)
        {
            for (int j = 0; j < jM; j++)
            {
                if (i == j)
                    M[i][j] = lambda + 2 * myu;
                else
                    M[i][j] = lambda;
            }
        }
        break;
    case 2:
        for (int i = 0; i < iM - 1; i++)
        {
            for (int j = 0; j < jM - 1; j++)
            {
                if (i == j)
                    M[i][j] = lambda + 2 * myu;
                else
                    M[i][j] = lambda;
            }
        }
        M[iM - 1][jM - 1] = 2 * myu;
        break;
    default:
        printf("Wrong input, matrix D\n");
        break;
    }
}

MatrixSchwarz operator*(MatrixSchwarz &N, MatrixSchwarz &L)
{
    MatrixSchwarz P(N.iM, L.jM);
    for (int i = 0; i < N.iM; i++)
    {
        for (int j = 0; j < L.jM; j++)
        {
            P.M[i][j] = 0;
            for (int k = 0; k < N.jM; k++)
            {
                P.M[i][j] += (N.M[i][k] * L.M[k][j]);
            }
        }
    }
    return P;
}

MatrixSchwarz operator*(const MatrixSchwarz &N, const double val)
{
    MatrixSchwarz P(N.iM, N.jM);
    for (int i = 0; i < N.iM; i++)
    {
        for (int j = 0; j < N.jM; j++)
        {
            P.M[i][j] = N.M[i][j] * val;
        }
    }
    return P;
}

MatrixSchwarz operator+(const MatrixSchwarz &N, const MatrixSchwarz &L)
{
    MatrixSchwarz P(N.iM, N.jM);
    for (int i = 0; i < N.iM; i++)
    {
        for (int j = 0; j < N.jM; j++)
        {
            P[i][j] = N.M[i][j] + L.M[i][j];
        }
    }
    return P;
}

void MatrixSchwarz::Create_Sy(strainMatrix &S, VectorSchwarz &m)
{
    int amntElements{0};
    vector<double> localNodes;
    vector<int> localElements;

    double A{0};
    vector<double> a;
    vector<double> b;
    vector<double> c;
    switch (S.dimTask)
    {
    case 1:
        amntElements = S.elements.GetSize();
        break;
    case 2:
        amntElements = S.elements.GetSize() / 3;
        break;
    }
    double h{0};
    Construct(S.iSize, amntElements);
    for (int j = 0; j < GetSize_j(); j++)
    {
        h = S.mesh.GetElement(j + 1) - S.mesh.GetElement(j);
        switch (S.dimTask)
        {
        case 1:
            M[0][j] = (m.GetElement(j + 1) - m.GetElement(j)) / h;
            M[1][j] = (m.GetElement(j + 1) + m.GetElement(j)) / (S.mesh.GetElement(j + 1) + S.mesh.GetElement(j));
            break;
        case 2:
            for (int i = 0; i < 3; i++)
                localElements.push_back(S.elements[j * 3 + i]);
            for (auto x : localElements)
                for (int i = 0; i < 2; i++)
                    localNodes.push_back(S.mesh[x * S.dimTask + i]);
            a.push_back(localNodes[2] * localNodes[5] - localNodes[4] * localNodes[3]);
            a.push_back(localNodes[4] * localNodes[1] - localNodes[5] * localNodes[0]);
            a.push_back(localNodes[0] * localNodes[3] - localNodes[2] * localNodes[1]);

            b.push_back(localNodes[3] - localNodes[5]);
            b.push_back(localNodes[5] - localNodes[1]);
            b.push_back(localNodes[1] - localNodes[3]);

            c.push_back(localNodes[4] - localNodes[2]);
            c.push_back(localNodes[0] - localNodes[4]);
            c.push_back(localNodes[2] - localNodes[0]);

            A = (1 / 2.0) * (localNodes[2] * localNodes[5] - localNodes[4] * localNodes[3] + localNodes[0] * localNodes[3] -
                             localNodes[0] * localNodes[5] + localNodes[4] * localNodes[1] - localNodes[2] * localNodes[1]);
            for (int i = 0; i < 3; i++)
            {
                M[0][j] += b[i] * m[i * S.dimTask] / (2 * A);
                M[1][j] += c[i] * m[i * S.dimTask + 1] / (2 * A);
                M[2][j] += (c[i] * m[i * S.dimTask]) / (2 * A) + (b[i] * m[i * S.dimTask + 1]) / (2 * A);
            }
            localElements.clear();
            localNodes.clear();
            break;
        }
    }
}

void MatrixSchwarz::Create_B(strainMatrix &S, basfuncMatrix &matrN)
{
    double h = matrN.arg.GetElement(matrN.numNode + 1) - matrN.arg.GetElement(matrN.numNode);
    Construct(S.iSize, S.dimTask * matrN.amntBE);
    for (int j = 0; j < GetSize_j(); j++)
    {
        switch (S.dimTask)
        {
        case 1:
            M[0][j] = (matrN.Get_N(matrN.node + h, j) - matrN.Get_N(matrN.node, j)) / h;
            M[1][j] = matrN.Get_N(matrN.node, j) / matrN.node;
            break;
        case 2:
            break;
        default:
            printf("Wrong input.\n");
            break;
        }
    }
}

void MatrixSchwarz::SetName(std::string str)
{
    name = str;
}

std::string MatrixSchwarz::GetName()
{
    return name;
}

void MatrixSchwarz::Record(std::string Route, int amntNodes, int amntSubdomains, double Coef)
{
    std::string sep = "_";
    std::string size = std::to_string(amntNodes - 1);
    std::string AS = std::to_string(amntSubdomains);

    if (amntNodes < 10)
    {
        size = "00" + size;
    }
    else if (amntNodes >= 10 && amntNodes < 100)
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
    std::ofstream outfile(Route);
    for (int j = 0; j < jM; j++)
    {
        for (int i = 0; i < iM; i++)
        {
            outfile << M[i][j] * Coef;
            outfile << " ";
        }
        outfile << std::endl;
    }
    outfile.close();
}

VectorSchwarz operator*(MatrixSchwarz &A, VectorSchwarz &b)
{
    VectorSchwarz y(b.GetSize());
    for (int i = 0; i < A.GetSize_i(); i++)
        for (int j = 0; j < A.GetSize_j(); j++)
            y[i] += A[i][j] * b[i];
    return y;
}

#endif
