#include <Windows.h>
#include <NuiApi.h>
#include <stdlib.h>

#include "OpenCVKinect.h"
#include "StairDetection.h"

#include <thread>

int main()
{
	OpenCVKinect mKinect;
	if (!mKinect.init())
	{
		std::cout << "Error initializing" << std::endl;
		return 1;
	}

	//mKinect.getData();
	//mKinect.getDepth8bit(depth);
	//color = mKinect.getColor();

	return 1;
}