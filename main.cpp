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

OpenCVKinect m_Kinect;
TextToSpeech m_tts;
//SignRecognizer m_sign;
//HumanFaceRecognizer m_face;
ObstacleDetection m_obstacle(800);

bool finished = false;

void KinectThread_Process()
{
	while (1) {
		if (finished)
			return;

		m_Kinect.updateData();
	}
}

void TextToSpeechThread_Process()
{
	m_tts.Initialize();

	while (1) {
		if (finished)
			return;

		m_tts.speak();
	}
}

void ObstacleDetectionThread_Process()
{
	cv::Mat colorImg, depthImg;
	uint64_t oldTimeStamp = 0, newTimeStamp = 0;

	while (waitKey(1) != 27) {
		if (finished)
			return;

		m_Kinect.getMatrix(m_Kinect.ColorDepth8bit, colorImg, Mat(), depthImg, newTimeStamp);
		if (newTimeStamp <= oldTimeStamp)
			continue;
		oldTimeStamp = newTimeStamp;
		
		cv::imshow("COLOR", colorImg);
		cv::imshow("DEPTH", depthImg);
	}
}

void FaceDetectionThread_Process()
{
	cv::Mat colorImg;
	uint64_t oldTimeStamp = 0, newTimeStamp = 0;

	while (waitKey(1) != 27) {
		if (finished)
			return;

		m_Kinect.getColor(colorImg, newTimeStamp);
		if (newTimeStamp <= oldTimeStamp)
			continue;
		oldTimeStamp = newTimeStamp;

		//m_face.runFaceRecognizer(&colorImg);
	}
}

void SignDetectionThread_Process()
{
	cv::Mat colorImg;
	uint64_t oldTimeStamp = 0, newTimeStamp = 0;

	while (waitKey(1) != 27) {
		if (finished)
			return;

		m_Kinect.getColor(colorImg, newTimeStamp);
		if (newTimeStamp <= oldTimeStamp)
			continue;
		oldTimeStamp = newTimeStamp;

		//m_sign.runRecognizer(colorImg);
	}
}

int main()
{
	if (!m_Kinect.init())
	{
		std::cerr << "Error initializing" << std::endl;
		return -1;
	}

	/// Ensure Matrix are filled before proceeding.
	uint64_t t = 0;
	do {
		m_Kinect.updateData();
		m_Kinect.getMatrix(m_Kinect.None, Mat(), Mat(), Mat(), t);
	} while (t == 0);

	/// Asynchronous threading
	auto KinectThread_Future = std::async(std::launch::async, KinectThread_Process);
	auto TextToSpeechThread_Future = std::async(std::launch::async, TextToSpeechThread_Process);
	auto ObstacleDetectionThread_Future = std::async(std::launch::async, ObstacleDetectionThread_Process);
	auto FaceDetectionThread_Future = std::async(std::launch::async, FaceDetectionThread_Process);
	auto SignDetectionThread_Future = std::async(std::launch::async, SignDetectionThread_Process);


	/// Idle Main thread to prevent from closing.
	/// Use getMatrix's time return to prevent over spam.
	uint64_t time = 0, oldtime = 0;
	while (waitKey(1) != 27) {
		m_Kinect.getMatrix(m_Kinect.None, Mat(), Mat(), Mat(), time);
		if (time <= oldtime)
			continue;

		imshow("Main Idle Window", Mat(100,100,CV_8U));
	}

	finished = true;
	KinectThread_Future.get();
	TextToSpeechThread_Future.get();
	ObstacleDetectionThread_Future.get();
	FaceDetectionThread_Future.get();
	SignDetectionThread_Future.get();
	return 1;
}

