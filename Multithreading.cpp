#include "Multithreading.h"

Multithreading::Multithreading()
	: m_obstacle(800)
{

}


Multithreading::~Multithreading()
{
	KinectThread_Future.get();
	TextToSpeechThread_Future.get();
	ObstacleDetectionThread_Future.get();
	FaceDetectionThread_Future.get();
	SignDetectionThread_Future.get();
}

bool Multithreading::InitializeKinect()
{
	if (!m_Kinect.init())
	{
		std::cerr << "Error initializing" << std::endl;
		return false;
	}

	/// Ensure Matrix are filled before proceeding.
	uint64_t t = 0;
	do {
		m_Kinect.updateData();
		m_Kinect.getMatrix(m_Kinect.None, Mat(), Mat(), Mat(), t);
	} while (t == 0);

	return true;
}

void Multithreading::CreateAsyncThreads()
{
	KinectThread_Future = std::async(std::launch::async, &Multithreading::KinectThread_Process, this);
	TextToSpeechThread_Future = std::async(std::launch::async, &Multithreading::TextToSpeechThread_Process, this);
	ObstacleDetectionThread_Future = std::async(std::launch::async, &Multithreading::ObstacleDetectionThread_Process, this);
	FaceDetectionThread_Future = std::async(std::launch::async, &Multithreading::FaceDetectionThread_Process, this);
	SignDetectionThread_Future = std::async(std::launch::async, &Multithreading::SignDetectionThread_Process, this);
}

void Multithreading::Hold()
{
	/// Idle Main thread to prevent from closing.
	/// Use getMatrix's time return to prevent over spam.
	uint64_t time = 0, oldtime = 0;
	while (waitKey(1) != 27) {
		m_Kinect.getMatrix(m_Kinect.None, Mat(), Mat(), Mat(), time);
		if (time <= oldtime)
			continue;

		//imshow("Main Idle Window", Mat(100, 100, CV_8U));
	}
}

void Multithreading::KinectThread_Process()
{
	while (1) {
		if (finished)
			return;

		m_Kinect.updateData();
	}
}

void Multithreading::TextToSpeechThread_Process()
{
	m_tts.Initialize();

	while (1) {
		if (finished)
			return;

		m_tts.speak();
	}
}

void Multithreading::ObstacleDetectionThread_Process()
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

void Multithreading::FaceDetectionThread_Process()
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

void Multithreading::SignDetectionThread_Process()
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
