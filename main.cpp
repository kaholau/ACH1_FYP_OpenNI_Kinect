#include <iostream>

#include "Multithreading.h"

int main()
{
	Multithreading mt;

	if (!mt.InitializeKinect())
		return -1;

	mt.CreateAsyncThreads();
	mt.Hold();

	return 1;
}

