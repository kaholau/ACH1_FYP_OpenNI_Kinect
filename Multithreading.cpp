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
	// Use getMatrix's time return to prevent over spam.
	uint64_t curTime = 0;
	namedWindow("Main Idle Window");
	while (waitKey(1) != ESCAPE_KEY) {
		curTime = cv::getTickCount() / cv::getTickFrequency();
		if (m_Kinect.recording && ((curTime - startRecordingTime) > recordingDuration))
		{
			m_Kinect.m_recorder.stop();
			std::cout << "stop recording" << std::endl;
			break;
		}
		m_Kinect.getMatrix(m_Kinect.None, Mat(), Mat(), Mat());
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

	while (waitKey(100) != ESCAPE_KEY) {
	//while (true) {
		if (finished)
			return;

		m_tts.speak();

	}
}

void on_trackbarCameraAngle(int i, void* userData)
{
	INuiSensor *pNuiSensor = ((INuiSensor*)userData);

	i = i - 27;
	if (userData!=NULL)
		pNuiSensor->NuiCameraElevationSetAngle((long)i);
	else
		std::cout << "NULL userdata" << std::endl;
}

void on_trackbarErosion(int i, void* userData)
{
	ObstacleDetection *obpt = ((ObstacleDetection*)userData);
	obpt->erosion_size = i;
}


void on_trackbarDilation1(int i, void* userData)
{
	ObstacleDetection *obpt = ((ObstacleDetection*)userData);
	obpt->dilation_size1 = i;
}

void on_trackbarDilation2(int i, void* userData)
{
	ObstacleDetection *obpt = ((ObstacleDetection*)userData);
	obpt->dilation_size2 = i;
}

void Multithreading::ObstacleDetectionThread_Process()
{

	cv::Mat colorImg, depth8bit, depthRaw;
	if (!m_Kinect.replay)
		m_Kinect.pNuiSensor->NuiCameraElevationSetAngle(m_obstacle.initCameraAngle);
	LONG angle = m_Kinect.getAngle();

	m_obstacle.setCameraAngle(angle);
	//namedWindow("DEPTH", CV_WINDOW_NORMAL);
	int track_angle = (int)angle;
	//createTrackbar("CameraAngle", "DEPTH", &track_angle, 23, on_trackbarCameraAngle, m_Kinect.pNuiSensor);
	
	int ero = 0;
	int di = 0;
	//createTrackbar("dilation1", "DEPTH", &di1, 21, on_trackbarDilation1, &m_obstacle);
	//createTrackbar("erosion", "DEPTH", &di2, 21, on_trackbarErosion, &m_obstacle);
	//createTrackbar("dilation2", "DEPTH", &di2, 21, on_trackbarDilation2, &m_obstacle);
	while (waitKey(1) != ESCAPE_KEY)
	{
		if (finished)
			return;
		double t = (double)getTickCount();
		
		m_Kinect.getMatrix(m_Kinect.All, colorImg, depthRaw, depth8bit);
		m_obstacle.setCameraAngle(m_Kinect.getAngle());

		m_obstacle.setCurrentColor(&colorImg);
		
		m_obstacle.SetCurrentRawDepth(&depthRaw);
		m_obstacle.run(&depth8bit);
		if (!m_Kinect.replay)
			m_obstacle.findHole(m_Kinect.pNuiSensor);
		m_obstacle.getOutputDepthImg(&depth8bit);
		m_obstacle.getOutputColorImg(&colorImg);

		/*t = 1/(((double)getTickCount() - t) / getTickFrequency());
		String fps = std::to_string(t) + "fps";
		putText(depth8bit, fps, Point(20,20), FONT_HERSHEY_PLAIN, 0.9, Scalar(128), 1);*/
		//std::cout << " Total used : " << t << " seconds" << std::endl;
		cv::imshow("DEPTH", depth8bit);
		Mat resizeColor = Mat(Size(320, 240), colorImg.type());
		resize(colorImg, resizeColor, Size(320, 240), 0, 0, 1);
		flip(resizeColor, resizeColor, 1);
		cv::imshow("resizeColor", resizeColor);
		//waitKey();
		//cv::imshow("COLOR", colorImg);
	}
}

void Multithreading::FaceDetectionThread_Process()
{
	cv::Mat colorImg;

	HANDLE handle = GetStdHandle(STD_INPUT_HANDLE);
	DWORD events;
	std::string faceName;

	while (waitKey(1) != ESCAPE_KEY) {
		if (finished)
			return;

		INPUT_RECORD buffer;
		PeekConsoleInput(handle, &buffer, 1, &events);
		if (events > 0 && !m_Kinect.recording)
		{
			ReadConsoleInput(handle, &buffer, 1, &events);
			switch (buffer.Event.KeyEvent.wVirtualKeyCode)
			{
			case 0x30:
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

			case 0x31:
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

		m_Kinect.getColor(colorImg);

		if (m_face.getisAddFace())
			m_face.addFace(colorImg);
		else
			m_face.runFaceRecognizer(&colorImg);

		cv::imshow("FACE DETECTION", colorImg);
	}


	//m_face.testExample();
}

void Multithreading::SignDetectionThread_Process()
{
	cv::Mat colorImg;

	while (waitKey(1) != ESCAPE_KEY) {
		if (finished)
			return;

		m_Kinect.getColor(colorImg);
		m_sign.setFrameSize(colorImg.cols, colorImg.rows);
		m_sign.runRecognizer(colorImg);
		cv::imshow("SIGN DETECTION", colorImg);
	}

	//m_sign.testExample();
}

void Multithreading::StairDetectionThread_Process()
{
	cv::Mat colorImg, depth8bit, depthRaw;
	std::vector<cv::Point> stairConvexHull;
	int previousFound = 0;
	int foundThreshold = 2;
	while (waitKey(1) != ESCAPE_KEY) {
		if (finished)
			return;

		m_Kinect.getMatrix(m_Kinect.ColorDepth8bit, colorImg, Mat(), depth8bit);
		m_stairs.Run(colorImg, depth8bit, stairConvexHull);
		if (!stairConvexHull.empty()) {
			if (previousFound > foundThreshold) {
				TextToSpeech::pushBack(string("Stairs Found"));
				//StairDetection::drawStairs("Stairs", colorImg, stairConvexHull);
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

