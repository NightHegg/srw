#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <vector>
#include <iostream>

template <class T>
class Vector
{
public:
	std::vector<double> V;
	std::vector<int> SchwarzNodes;
	int LeftBoundary;
	int RightBoundary;
	bool UsingMethodSchwarz;
	Vector() = default;
	Vector(int i)
	{
		iV = i;
		V = new double[iV];

		NullVector();

		SchwarzNodes = std::vector<int>();
		LeftBoundary = 0;
		RightBoundary = iV - 1;
		UsingMethodSchwarz = false;
	}
	void NullVector()
	{
		for (int i = 0; i < iV; i++)
		{
			this->V[i] = 0;
		}
	}
	~Vector()
	{
		delete[] V;
	}
	void Construct(int i)
	{
		this->iV = i;
		V = new double[iV];
		LeftBoundary = 0;
		RightBoundary = iV - 1;
		UsingMethodSchwarz = false;
	}
	Vector(const Vector& N)
	{
		Construct(N.iV);
		for (int i = 0; i < iV; i++)
		{
			V[i] = N.V[i];
		}
		LeftBoundary = N.LeftBoundary;
		RightBoundary = N.RightBoundary;
		UsingMethodSchwarz = 0;
	}
	void operator= (Vector N)
	{
		Construct(N.iV);
		for (int i = 0; i < iV; i++)
		{
			V[i] = N.V[i];
		}
		LeftBoundary = N.LeftBoundary;
		RightBoundary = N.RightBoundary;
	}
	void Show()
	{
		for (int i = 0; i < iV; i++)
		{
			printf("%g %d\n", V[i], i);
		}
	}
	double &operator [] (int _i)
	{
		return V[_i];
	}
	double operator-(double i)
	{
		return V[iV] - i;
	}
	double operator-(Vector A)
	{
		return V[iV] - A[A.iV];
	}
	
	void FillVector(double var)
	{
		for (int i = 0;i < iV; i++)
		{
			this->V[i] = var;
		}
	}
	
};

#endif