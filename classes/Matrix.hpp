#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
/**
 * Matrix class with some basic features
 * TODO Add getters and private section // DONE
 * ? Do I need to add template for this class?
 * ? Do I need to create child class? // DONE
 */

class Matrix
{
protected:
    double **M;
    int iM, jM;
public:
    Matrix() = default;
    Matrix(int i, int j);
    Matrix(const Matrix &N);
    void Construct(int i, int j);
    ~Matrix() = default;
    void Null();
    void Show();
    void Transpose(Matrix &A);
    void Identity();
    void Inverse(Matrix &A);
    double *operator[](const int index);
    friend Matrix operator*(const Matrix &N, const Matrix &L);
    friend Matrix operator+(const Matrix &N, const Matrix &L);
    friend Matrix operator*(const Matrix &N, const double val);
    Matrix operator=(const Matrix &N);
    int GetSize_i();
    int GetSize_j();
    double GetElement(int i, int j);
    void SetElement(int i, int j, double a);
};

Matrix::Matrix(int i, int j)
{
    Construct(i, j);
}

void Matrix::Construct(int i, int j)
{
    this->iM = i;
    this->jM = j;
    M = new double *[iM];
    for (int i = 0; i < iM; i++)
        M[i] = new double[jM];

    Null();
}

void Matrix::Null()
{
    for (int i = 0; i < iM; i++)
    {
        for (int j = 0; j < jM; j++)
        {
            this->M[i][j] = 0;
        }
    }
}
Matrix::Matrix(const Matrix &N)
{
    Construct(N.iM, N.jM);
    for (int i = 0; i < iM; i++)
    {
        for (int j = 0; j < jM; j++)
        {
            this->M[i][j] = N.M[i][j];
        }
    }
}

void Matrix::Show()
{
    std::cout << std::endl;
    for (int i = 0; i < iM; i++)
    {
        for (int j = 0; j < jM; j++)
        {
            std::cout.width(10);
            std::cout << M[i][j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void Matrix::Transpose(Matrix &A)
{
    A.Construct(jM, iM);
    for (int i = 0; i < A.iM; i++)
    {
        for (int j = 0; j < A.jM; j++)
        {
            A.M[i][j] = this->M[j][i];
        }
    }
}

void Matrix::Identity()
{
    for (int i = 0; i < iM; i++)
    {
        for (int j = 0; j < jM; j++)
        {
            if (i == j)
            {
                M[i][j] = 1;
            }
            M[i][j] = 0;
        }
    }
}

void Matrix::Inverse(Matrix &A)
{
    double Buf{0};
    A.Construct(iM, jM);
    A.Identity();
    for (int i = 0; i < iM; i++)
    {
        Buf = M[i][i];
        for (int j = 0; j < jM; j++)
        {
            M[i][j] /= Buf * 1.0;
            A.M[i][j] /= Buf * 1.0;
        }
        for (int k = i + 1; k < iM; k++)
        {
            Buf = M[k][i];
            for (int j = 0; j < jM; j++)
            {
                M[k][j] -= M[i][j] * Buf;
                A.M[k][j] -= A.M[i][j] * Buf;
            }
        }
    }
    for (int i = iM - 1; i > 0; i--)
    {
        for (int k = i - 1; k >= 0; k--)
        {
            Buf = M[k][i];
            for (int j = 0; j < jM; j++)
            {
                M[k][j] -= M[i][j] * Buf;
                A.M[k][j] -= A.M[i][j] * Buf;
            }
        }
    }
}

Matrix operator+(const Matrix &N, const Matrix &L)
{
    Matrix P(N.iM, N.jM);
    for (int i = 0; i < N.iM; i++)
    {
        for (int j = 0; j < N.jM; j++)
        {
            P[i][j]=N.M[i][j]+L.M[i][j];
        }
    }
    return P;
}

Matrix operator*(const Matrix &N, const Matrix &L)
{
    Matrix P(N.iM, L.jM);
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

Matrix operator*(const Matrix &N, const double val)
{
    Matrix P(N.iM, N.jM);
    for (int i = 0; i < N.iM; i++)
    {
        for (int j = 0; j < N.jM; j++)
        {
            P.M[i][j] = N.M[i][j] * val;
        }
    }
    return P;
}

Matrix Matrix::operator=(const Matrix &N)
{
    Construct(N.iM, N.jM);
    for (int i = 0; i < iM; i++)
    {
        for (int j = 0; j < jM; j++)
        {
            this->M[i][j] = N.M[i][j];
        }
    }
    return *this;
}

double *Matrix::operator[](const int index)
{
    return M[index];
}

int Matrix::GetSize_i()
{
    return iM;
}

int Matrix::GetSize_j()
{
    return jM;
}

double Matrix::GetElement(int i, int j)
{
    return M[i][j];
}

void Matrix::SetElement(int i, int j, double a)
{
    M[i][j]=a;
}

#endif
