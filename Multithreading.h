#pragma once
#include <Windows.h>
#include <NuiApi.h>
#include <stdlib.h>
#include <thread>
#include <future>

#include "include.h"
#include "OpenCVKinect.h"


class Multithreading
{
	OpenCVKinect m_Kinect;
	TextToSpeech m_tts;
	SignRecognizer m_sign;
	HumanFaceRecognizer m_face;
	ObstacleDetection m_obstacle;
	StairDetection m_stairs;
	bool finished = false;
	uint64_t startRecordingTime = 0;
	const uint64_t recordingDuration = 20 ; //second

	std::future<void> KinectThread_Future;
	std::future<void> TextToSpeechThread_Future;
	std::future<void> ObstacleDetectionThread_Future;
	std::future<void> StairDetectionThread_Future;
	std::future<void> FaceDetectionThread_Future;
	std::future<void> SignDetectionThread_Future;

	void KinectThread_Process();
	void TextToSpeechThread_Process();
	void ObstacleDetectionThread_Process();
	void StairDetectionThread_Process();
	void FaceDetectionThread_Process();
	void SignDetectionThread_Process();

	bool initialized = false;

	std::ofstream FilePath;
	std::ofstream FileStair;
	std::ofstream FileFace;
	std::ofstream FileSign;
	bool recordProcessTime = true;
public:
	Multithreading();
	~Multithreading();

	bool InitializeKinect();
	void CreateAsyncThreads();
	void Hold();

	//void on_trackbar(int, void*);
};

