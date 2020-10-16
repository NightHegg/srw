#include <vector>
#include <iostream>
#include "Body.h"

using namespace std;

int main()
{
	int count{1}; //dimTask, amntNodes(1), amntSubdomains (Schwarz attribute), stopCriteria
	vector<vector<double>> solSet = {
		{2, 1}};
	printf("# Planned: %d calculations\n", solSet.size());
	printf("//\n//\n//\n");
	for (auto i : solSet)
	{
		printf("# Running: %d of %d calculation...\n", count++, solSet.size());
		printf("//\n//\n//\n");
		Solve(i);
	}
	printf("//\n//\n//\n");
	printf("# End of iterations");
}