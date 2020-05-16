#include <vector>
#include <iostream>
#include "Body.h"

using namespace std;

int main()
{
	vector<vector<double>> solSet = {
		{1, 50, 1},
		{1, 100, 1},
		{1, 200, 1},
		{1, 50, 2, 1e-5}};

	for (auto i : solSet)
	{
		Solve(i);
	}
}