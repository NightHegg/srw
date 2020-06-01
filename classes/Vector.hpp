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
	double NormM();
	double NormL();
	double NormEuclidean();
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
		printf("%4g\n", V[i]);
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

double Vector::NormM()
{
	double max{0};
	for (int i = 0; i < iV; i++)
		if (abs(V[i]) > max)
			max = abs(V[i]);
	return max;
}

double Vector::NormL()
{
	double sum{0};
	for (int i = 0; i < iV; i++)
	{
		sum += abs(V[i]);
	}
	return sum;
}

double Vector::NormEuclidean()
{
	double sum{0};
	for (int i = 0; i < iV; i++)
	{
		sum += pow(V[i],2);
	}
	return sqrt(sum);
}

#endif