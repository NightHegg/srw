#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <vector>
#include <iostream>

class Vector
{
protected:
	double *V;
	int iV;

public:
	Vector() = default;
	Vector(int index);
	~Vector() = default;
	Vector(const Vector &N);
	void Null();
	void Construct(int i);
	void Show();
	void Fill(double var);
	Vector &operator=(const Vector &N)
	{
		Construct(N.iV);
		for (int i = 0; i < iV; i++)
		{
			V[i] = N.V[i];
		}
		return *this;
	}
	double &operator[](int ind);
	int GetSize();
	double GetElement(int i);
	void SetElement(int i, double a);
};

Vector::Vector(int index)
{
	Construct(index);
	Null();
}

void Vector::Null()
{
	for (int i = 0; i < iV; i++)
	{
		V[i] = 0;
	}
}

void Vector::Construct(int i)
{
	this->iV = i;
	V = new double[iV];
	for (int i = 0; i < iV; i++)
		V[i] = 0;
}

Vector::Vector(const Vector &N)
{
	Construct(N.iV);
	for (int i = 0; i < iV; i++)
	{
		V[i] = N.V[i];
	}
}

void Vector::Show()
{
	for (int i = 0; i < iV; i++)
	{
		printf("%g\n", V[i]);
	}
}

void Vector::Fill(double var)
{
	for (int i = 0; i < iV; i++)
	{
		this->V[i] = var;
	}
}

double &Vector::operator[](int ind)
{
	return V[ind];
}

int Vector::GetSize()
{
	return iV;
}
double Vector::GetElement(int i)
{
	return V[i];
}

void Vector::SetElement(int i, double a)
{
	V[i] = a;
}

#endif