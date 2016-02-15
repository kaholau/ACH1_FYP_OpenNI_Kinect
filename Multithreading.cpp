#include "Multithreading.h"

Multithreading::Multithreading()
	: m_obstacle(1020)
{

}


Multithreading::~Multithreading()
{
	finished = true;
	KinectThread_Future.get();
	TextToSpeechThread_Future.get();
	ObstacleDetectionThread_Future.get();
	//FaceDetectionThread_Future.get();
	//SignDetectionThread_Future.get();
	StairDetectionThread_Future.get();
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

	/*if (m_Kinect.recording) {
		m_Kinect.m_recorder.start();

		m_Kinect.pNuiSensor->NuiCameraElevationSetAngle(0);
	}*/


	return true;
}

void Multithreading::CreateAsyncThreads()
{
	KinectThread_Future = std::async(std::launch::async, &Multithreading::KinectThread_Process, this);
	/*if (m_Kinect.recording)
		return;*/

	TextToSpeechThread_Future = std::async(std::launch::async, &Multithreading::TextToSpeechThread_Process, this);
	ObstacleDetectionThread_Future = std::async(std::launch::async, &Multithreading::ObstacleDetectionThread_Process, this);
	//FaceDetectionThread_Future = std::async(std::launch::async, &Multithreading::FaceDetectionThread_Process, this);
	//SignDetectionThread_Future = std::async(std::launch::async, &Multithreading::SignDetectionThread_Process, this);
	StairDetectionThread_Future = std::async(std::launch::async, &Multithreading::StairDetectionThread_Process, this);
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
	uint64_t oldTimeStamp = 0, newTimeStamp = 0;

	while (1) {
		if (finished)
			return;

		m_Kinect.updateData();
		m_Kinect.getMatrix(m_Kinect.ColorDepth8bit, colorImg, Mat(), depth8bit, newTimeStamp);

		//cv::imshow("ORIGINAL COLOR", colorImg);
		cv::imshow("ORIGINAL DEPTH", depth8bit);
		cv::waitKey(1);
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

void on_trackbarCameraAngle(int i, void* userData)
{
	INuiSensor *pNuiSensor = ((INuiSensor*)userData);

	i = i - 18;
	if (userData!=NULL)
		pNuiSensor->NuiCameraElevationSetAngle((long)i);
	else
		std::cout << "NULL userdata" << std::endl;
}

void on_trackbarDilation(int i, void* userData)
{
	ObstacleDetection *obpt = ((ObstacleDetection*)userData);
	obpt->dilation_size = i;
}

void on_trackbarErosion(int i, void* userData)
{
	ObstacleDetection *obpt = ((ObstacleDetection*)userData);
	obpt->erosion_size = i;
}

void Multithreading::ObstacleDetectionThread_Process()
{

	cv::Mat colorImg, depth8bit, depthRaw;
	uint64_t oldTimeStamp = 0, newTimeStamp = 0;
	LONG angle = m_Kinect.getAngle();

	m_obstacle.setCameraAngle(angle);
	namedWindow("DEPTH", CV_WINDOW_NORMAL);
	int track_angle = (int)angle;
	createTrackbar("CameraAngle", "DEPTH", &track_angle, 23, on_trackbarCameraAngle, m_Kinect.pNuiSensor);
	
	int ero = 0;
	int di= 0;
	createTrackbar("erosion", "DEPTH", &ero, 21, on_trackbarErosion, &m_obstacle);
	createTrackbar("dilation", "DEPTH", &di, 21, on_trackbarDilation, &m_obstacle);
	while (waitKey(1) != 27) 
	{
		if (finished)
			return;
		double t = (double)getTickCount();
		
		m_Kinect.getMatrix(m_Kinect.All, colorImg, depthRaw, depth8bit, newTimeStamp);
		if (newTimeStamp <= oldTimeStamp)
			continue;
		oldTimeStamp = newTimeStamp;

		m_obstacle.setCurrentColor(&colorImg);
		m_obstacle.setCameraAngle(m_Kinect.getAngle());
		m_obstacle.SetCurrentRawDepth(&depthRaw);
		m_obstacle.run(&depth8bit);
		m_obstacle.getOutputDepthImg(&depth8bit);
		m_obstacle.getOutputColorImg(&colorImg);

		t = 1/(((double)getTickCount() - t) / getTickFrequency());
		String fps = std::to_string(t) + "fps";
		putText(depth8bit, fps, Point(20,20), FONT_HERSHEY_PLAIN, 0.9, Scalar(128), 1);
		//std::cout << " Total used : " << t << " seconds" << std::endl;
		cv::imshow("DEPTH", depth8bit);
		//waitKey();
		//cv::imshow("COLOR", colorImg);
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
		m_sign.setFrameSize(colorImg.cols, colorImg.rows);
		m_sign.runRecognizer(colorImg);
		cv::imshow("SIGN DETECTION", colorImg);
	}
}

void Multithreading::StairDetectionThread_Process()
{
	cv::Mat colorImg, depth8bit, depthRaw;
	uint64_t oldTimeStamp = 0, newTimeStamp = 0;
	std::vector<cv::Point> stairConvexHull;

	while (waitKey(1) != 27) {
		if (finished)
			return;

		m_Kinect.getMatrix(m_Kinect.ColorDepth8bit, colorImg, Mat(), depth8bit, newTimeStamp);
		if (newTimeStamp <= oldTimeStamp)
			continue;
		oldTimeStamp = newTimeStamp;

		m_stairs.Run(colorImg, depth8bit, stairConvexHull);
		StairDetection::drawStairs("Stairs", colorImg, stairConvexHull);
		stairConvexHull.clear();
	}
}