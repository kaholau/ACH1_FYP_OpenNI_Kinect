#include "Multithreading.h"

Multithreading::Multithreading()
	: m_obstacle(800)
{

}


Multithreading::~Multithreading()
{
	finished = true;
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
	cv::Mat colorImg, depth8bit;
	uint64_t t = 0;
	while (1) {
		if (finished)
			return;

		m_Kinect.updateData();
		//m_Kinect.getMatrix(m_Kinect.ColorDepth8bit, colorImg, Mat(), depth8bit, t);

		//cv::imshow("ORIGINAL COLOR", colorImg);
		//cv::imshow("ORIGINAL DEPTH", depth8bit);
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
	cv::Mat colorImg, depth8bit, depthRaw;
	uint64_t oldTimeStamp = 0, newTimeStamp = 0;

	while (waitKey(1) != 27) {
		if (finished)
			return;

		m_Kinect.getMatrix(m_Kinect.All, colorImg, depthRaw, depth8bit, newTimeStamp);
		if (newTimeStamp <= oldTimeStamp)
			continue;
		oldTimeStamp = newTimeStamp;

		m_obstacle.getCurrentColor(&colorImg);
		m_obstacle.setCameraAngle(0);
		m_obstacle.SetCurrentRawDepth(&depthRaw);
		m_obstacle.run(&depth8bit);
		m_obstacle.getOutputDepthImg(&depth8bit);
		m_obstacle.getOutputColorImg(&colorImg);

		cv::imshow("DEPTH", depth8bit);
		cv::imshow("COLOR", colorImg);
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

		m_face.runFaceRecognizer(&colorImg);

		cv::imshow("FACE DETECTION", colorImg);
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

		m_sign.runRecognizer(colorImg);
		cv::imshow("SIGN DETECTION", colorImg);
	}
}
