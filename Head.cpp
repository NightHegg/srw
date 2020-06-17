#include <vector>
#include <iostream>
#include "Body.h"

using namespace std;

int main()
{
	int count{1}; //dimTask, amntNodes(1), schwarz, stopCriteria
	vector<vector<double>> solSet = {
		{2,1}};

	printf("# Planned: %d calculations\n\n", solSet.size());
	for (auto i : solSet)
	{
		printf("# Running: %d of %d calculation...\n", count++, solSet.size());
		Solve(i);
	}
	printf("# End of iterations");
}