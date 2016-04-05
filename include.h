#pragma once

#include <iostream>

#include "detector.h"
#include "HumanFaceRecognizer.h"
//#include "Multithreading.h"
#include "ObstacleDetection.h"
//#include "OpenCVKinect.h"
#include "SerialClass.h"
#include "SignRecognizer.h"
#include "StairDetection.h"
#include "TextToSpeech.h"


#define COLOUR_FRAME_USE_HIGHEST_RESOLUTION_1280x960

#ifdef COLOUR_FRAME_USE_HIGHEST_RESOLUTION_1280x960
	#define COLOUR_FRAME_1280x960
	#define COLOUR_FRAME_WIDTH     1280
	#define COLOUR_FRAME_HEIGHT    960
#else
	#define COLOUR_FRAME_640x480
	#define COLOUR_FRAME_WIDTH     640
	#define COLOUR_FRAME_HEIGHT    480
#endif




/* Struct */
typedef struct DetectionInfo
{
	bool isRecognized;
	int label;
	cv::Point centerPos;
	cv::Size size;
	//short counter[NUM_OF_PERSON + 1];
	std::vector<int> counter;
	short undetected_counter;
} DetectionInfo;

