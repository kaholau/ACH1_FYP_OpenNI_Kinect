#pragma once
#include <Windows.h>
#include <NuiApi.h>
#include <stdlib.h>

#include "OpenCVKinect.h"
#include "TextToSpeech.h"
#include "SignRecognizer.h"
#include "HumanFaceRecognizer.h"
#include "ObstacleDetection.h"

#include <thread>
#include <future>
#include <iostream>

class Multithreading
{
	OpenCVKinect m_Kinect;
	TextToSpeech m_tts;
	SignRecognizer m_sign;
	HumanFaceRecognizer m_face;
	ObstacleDetection m_obstacle;
	bool finished = false;

	std::future<void> KinectThread_Future;
	std::future<void> TextToSpeechThread_Future;
	std::future<void> ObstacleDetectionThread_Future;
	std::future<void> FaceDetectionThread_Future;
	std::future<void> SignDetectionThread_Future;

	void KinectThread_Process();
	void TextToSpeechThread_Process();
	void ObstacleDetectionThread_Process();
	void FaceDetectionThread_Process();
	void SignDetectionThread_Process();

public:
	Multithreading();
	~Multithreading();

	bool InitializeKinect();
	void CreateAsyncThreads();
	void Hold();
};

