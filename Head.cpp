#include <vector>
#include <iostream>
#include "Body.h"

using namespace std;

int main()
{
	int count{1};
	vector<vector<double>> solSet = {
		{1, 50, 1}};

	printf("Planned: %d calculations\n\n", solSet.size());
	for (auto i : solSet)
	{
		printf("Running %d iteration...\n\n", count++);
		Solve(i);
	}
	printf("End of iterations\n");
}