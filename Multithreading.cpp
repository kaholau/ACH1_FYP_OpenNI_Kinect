#include "Multithreading.h"

Multithreading::Multithreading()
//kaho wear kinect at 1020mm
//color lab table give 820mm
//yumi wear kinect at 900mm
	: m_obstacle(1020)
{

}


Multithreading::~Multithreading()
{
	finished = true;

	if (!initialized)
		return; 

	if (m_Kinect.recording)
		return;

	KinectThread_Future.get();
	TextToSpeechThread_Future.get();
	ObstacleDetectionThread_Future.get();
	FaceDetectionThread_Future.get();
	SignDetectionThread_Future.get();
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
	do {
		m_Kinect.updateData();
	} while (m_Kinect.getTimestamp() == -1);

	if (m_Kinect.recording) {
		m_Kinect.m_recorder.start();
		m_Kinect.pNuiSensor->NuiCameraElevationSetAngle(m_obstacle.initCameraAngle);

		startRecordingTime = cv::getTickCount() / cv::getTickFrequency();
		std::cout << "start recording" << std::endl;
	}

	initialized = true;
	return true;
}

void Multithreading::CreateAsyncThreads()
{
	KinectThread_Future = std::async(std::launch::async, &Multithreading::KinectThread_Process, this);
	if (m_Kinect.recording)
		return;

	TextToSpeechThread_Future = std::async(std::launch::async, &Multithreading::TextToSpeechThread_Process, this);
	ObstacleDetectionThread_Future = std::async(std::launch::async, &Multithreading::ObstacleDetectionThread_Process, this);
	FaceDetectionThread_Future = std::async(std::launch::async, &Multithreading::FaceDetectionThread_Process, this);
	SignDetectionThread_Future = std::async(std::launch::async, &Multithreading::SignDetectionThread_Process, this);
	StairDetectionThread_Future = std::async(std::launch::async, &Multithreading::StairDetectionThread_Process, this);
}


/* Idle Main thread to prevent from closing */
void Multithreading::Hold()
{
	HANDLE handle = GetStdHandle(STD_INPUT_HANDLE);
	DWORD events;
	std::string faceName;

	// Use getMatrix's time return to prevent over spam.
	uint64_t curTime = 0;
	namedWindow("Main Idle Window");
	imshow("Main Idle Window", cv::Mat(cv::Size(50, 50), CV_8UC3));
	while (waitKey(1) != ESCAPE_KEY) {
		curTime = cv::getTickCount() / cv::getTickFrequency();
		if (m_Kinect.recording && ((curTime - startRecordingTime) > recordingDuration))
		{
			m_Kinect.m_recorder.stop();
			std::cout << "stop recording" << std::endl;
			break;
		}
		m_Kinect.getMatrix(m_Kinect.None, Mat(), Mat(), Mat());


		INPUT_RECORD buffer;
		PeekConsoleInput(handle, &buffer, 1, &events);
		if (events > 0 && !m_Kinect.recording)
		{
			ReadConsoleInput(handle, &buffer, 1, &events);
			switch (buffer.Event.KeyEvent.wVirtualKeyCode)
			{
			case 0x41:
				if (!m_face.getisAddFace())
				{
					while (true)
					{
						std::cout << "Please enter person's name to be added: ";
						std::getline(std::cin, faceName);
						if ((strcmp(faceName.c_str(), "") != 0) &&
							(strcmp(faceName.c_str(), "0") != 0) &&
							(strcmp(faceName.c_str(), "1") != 0))
						{
							m_face.setNameStr(faceName);
							break;
						}
					}
					m_face.setisAddFace(true);
					m_face.setisUpdated(true);
				}
				break;

			case 0x53:
				m_face.clearNameStr();
				m_face.setisAddFace(false);
				break;

			case 0x4F:
				std::cout << "Play" << cv::getTickCount() << std::endl;
				m_Kinect.setPlayspeed(1);
				break;
			case 0x50:
				std::cout << "Pause" << cv::getTickCount() << std::endl;
				m_Kinect.setPlayspeed(-1);
				break;
			}
		}
	}

	std::cout << "Exit Program!" << std::endl;
	finished = true;
	if (m_face.getisUpdated())
	{
		m_face.saveFaceDatabase();
		std::cout << "Face Database Saved." << std::endl;
	}
}

void Multithreading::KinectThread_Process()
{
	cv::Mat colorImg, depth8bit;

	while (1) {
		if (finished)
			return;

		m_Kinect.updateData();
		m_Kinect.getMatrix(m_Kinect.ColorDepth8bit, colorImg, Mat(), depth8bit);
		
		//cv::imshow("ORIGINAL COLOR", colorImg);

		//cv::imshow("ORIGINAL DEPTH", depth8bit);
		//cv::waitKey(1);
	}
}

void Multithreading::TextToSpeechThread_Process()
{
	m_tts.Initialize();

	while (waitKey(1) != ESCAPE_KEY) {
		if (finished)
			return;

		m_tts.speak();
	}
}


void Multithreading::ObstacleDetectionThread_Process()
{

	cv::Mat colorImg, depth8bit, depthRaw;
	if (!m_obstacle.test)
	{
		if (!m_Kinect.replay)
			m_Kinect.pNuiSensor->NuiCameraElevationSetAngle(m_obstacle.initCameraAngle);
		m_Kinect.getMatrix(m_Kinect.All, colorImg, depthRaw, depth8bit);
		m_obstacle.init(depthRaw.size());
		while (waitKey(1) != ESCAPE_KEY)
		{
			if (finished)
				return;

			if (m_face.getisAddFace())
				continue;

			double t = (double)getTickCount();
			m_Kinect.getMatrix(m_Kinect.All, colorImg, depthRaw, depth8bit);

			m_obstacle.run(&depth8bit, &depthRaw, m_Kinect.getAngle());
			if (!m_Kinect.replay)
			{
				int angle = m_obstacle.findHole();
				if (angle != 99)
					m_Kinect.pNuiSensor->NuiCameraElevationSetAngle(angle);

			}

			m_obstacle.getOutputDepthImg(&depth8bit);

			t = 1 / (((double)getTickCount() - t) / getTickFrequency());
			String fps = std::to_string(t) + "fps";
			putText(depth8bit, fps, Point(20, 20), FONT_HERSHEY_PLAIN, 0.9, Scalar(128), 1);
			cv::imshow("DEPTH", depth8bit);
		}
	}
	else
	{
		m_obstacle.init(Size(320,240));
		string colorFile = "132_colour.png";
		string depthFile = "132_depthraw.png";
		colorImg = imread(colorFile, CV_LOAD_IMAGE_COLOR);
	//	cv::cvtColor(temp, colorImg, CV_BGR2RGB);
		depthRaw = imread(depthFile, CV_LOAD_IMAGE_ANYDEPTH);
		double min, max;
		cv::minMaxLoc(depthRaw, &min, &max);
		depthRaw.convertTo(depth8bit, CV_8U, 255.0 / max);

		m_obstacle.run(&depth8bit, &depthRaw, -17*2);
		m_obstacle.getOutputDepthImg(&depth8bit);
		cv::imshow("DEPTH", depth8bit);
		resize(colorImg, colorImg, Size(320, 240), 0, 0, 1);
		cv::imshow("colorImg", colorImg);
		waitKey();
		//cv::imshow("COLOR", colorImg);
	}
	
}

void Multithreading::FaceDetectionThread_Process()
{
#ifndef TEST_FACE
#ifdef EXTRACT_FRAME_FOR_FACE
	int image_num = 0;
#endif
	cv::Mat colorImg;

	while (waitKey(1) != ESCAPE_KEY) {
		if (finished)
			return;


#ifndef EXTRACT_FRAME_FOR_FACE
		m_Kinect.getColor(colorImg);

		if (m_face.getisAddFace())
			m_face.addFace(colorImg);
		else
			m_face.runFaceRecognizer(&colorImg);

		cv::imshow("FACE DETECTION", colorImg);

#else
		m_Kinect.getColor(colorImg);

		if (!colorImg.empty())
		{
			std::stringstream oss;
			oss.str("");
			oss << IMAGE_DIR << image_num++ << IMAGE_NAME_POSTFIX << IMAGE_EXTENSION;
			cv::imwrite(oss.str(), colorImg);

			colorImg.release();
			waitKey(80);
		}
#endif
	}
#else
	m_face.testExample();
#endif
}

void Multithreading::SignDetectionThread_Process()
{
	cv::Mat colorImg;
	namedWindow("Sign Detection");
	imshow("Sign Detection", cv::Mat(cv::Size(320, 240), CV_8UC3));
	while (waitKey(20) != ESCAPE_KEY) {
		if (finished)
			return;

		if (m_face.getisAddFace())
			continue;

		m_Kinect.getColor(colorImg);
		m_sign.setFrameSize(colorImg.cols, colorImg.rows);
		m_sign.runRecognizer(colorImg);

		//cv::imshow("SIGN DETECTION", colorImg);
	}

	//m_sign.testExample();
}

void Multithreading::StairDetectionThread_Process()
{
	cv::Mat colorImg, depth8bit, depthRaw;
	std::vector<cv::Point> stairConvexHull;
	int previousFound = 0;
	int foundThreshold = 2;
	cv::namedWindow("Stairs");
	while (waitKey(1) != ESCAPE_KEY) {
		if (finished)
			return;

		if (m_face.getisAddFace())
			continue;

		m_Kinect.getMatrix(m_Kinect.ColorDepth8bit, colorImg, Mat(), depth8bit);
		m_stairs.Run(colorImg, depth8bit, stairConvexHull);
		if (!stairConvexHull.empty()) {
			if (previousFound >= foundThreshold) {
				if (ObstacleDetection::angleSetToLookDown)
					TextToSpeech::pushBack(string("Downstairs found"));
				else
					TextToSpeech::pushBack(string("Upstairs found"));

				StairDetection::drawStairs("Stairs", colorImg, stairConvexHull);
				++previousFound;
			}
			else {
				++previousFound;
			}
		}
		else
		{
			--previousFound;
			if (previousFound < 0)
				previousFound = 0;
		}
		stairConvexHull.clear();
	}
}

