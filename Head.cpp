#include <vector>
#include <iostream>
#include "Body.h"

using namespace std;

int main()
{
	int count{1};
	vector<vector<double>> solSet = {
		{1, 100, 2, 1e-5},
		{1, 100, 4, 1e-5},
		{1, 100, 10, 1e-5},
		{1, 200, 2, 1e-5},
		{1, 200, 4, 1e-5},
		{1, 200, 10, 1e-5}};

	printf("Planned: %d calculations\n\n", solSet.size());
	for (auto i : solSet)
	{
		printf("Running %d iteration...\n\n", count++);
		Solve(i);
	}
	printf("End of iterations\n");
}