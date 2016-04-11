#include "Multithreading.h"

Multithreading::Multithreading()
//kaho wear kinect at 1020mm
//color lab table give 820mm
	: m_obstacle(1020)
{

}


Multithreading::~Multithreading()
{
	finished = true;
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
	uint64_t t = 0;
	do {
		m_Kinect.updateData();
		m_Kinect.getMatrix(m_Kinect.None, Mat(), Mat(), Mat(), t);
	} while (t == 0);

	if (m_Kinect.recording) {
		m_Kinect.m_recorder.start();
		m_Kinect.pNuiSensor->NuiCameraElevationSetAngle(m_obstacle.initCameraAngle);

		startRecordingTime = cv::getTickCount() / cv::getTickFrequency();
		std::cout << "start recording" << std::endl;
	}

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

void Multithreading::Hold()
{
	// Idle Main thread to prevent from closing.
	// Use getMatrix's time return to prevent over spam.
	uint64_t time = 0, oldtime = 0;
	uint64_t curTime = 0;
	while (waitKey(1) != ESCAPE_KEY) {
		curTime = cv::getTickCount() / cv::getTickFrequency();
		if (m_Kinect.recording && ((curTime - startRecordingTime) > recordingDuration))
		{
			m_Kinect.m_recorder.stop();
			std::cout << "stop recording" << std::endl;
			break;
		}
		m_Kinect.getMatrix(m_Kinect.None, Mat(), Mat(), Mat(), time);
		if (time <= oldtime)
			continue;
		
		imshow("Main Idle Window", Mat(100, 100, CV_8U));
	}

	std::cout << "Exit Program!" << std::endl;
	finished = true;
	if (m_face.isUpdated)
	{
		m_face.model->save(DB_FACE_FILE_PATH);
		std::ofstream fout;
		fout.open(DB_NAME_FILE_PATH, std::ios::out);
		for (int i = 0; i < m_face.PERSON_NAME.size(); i++)
		{
			fout << i << ',' << m_face.PERSON_NAME[i] << std::endl;
		}

		std::cout << "Face Database Saved." << std::endl;
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
		//cv::imshow("ORIGINAL DEPTH", depth8bit);
		//cv::waitKey(1);
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
	uint64_t oldTimeStamp = 0, newTimeStamp = 0;
	if (!m_Kinect.replay)
		m_Kinect.pNuiSensor->NuiCameraElevationSetAngle(m_obstacle.initCameraAngle);
	LONG angle = m_Kinect.getAngle();

	m_obstacle.setCameraAngle(angle);
	namedWindow("DEPTH", CV_WINDOW_NORMAL);
	int track_angle = (int)angle;
	createTrackbar("CameraAngle", "DEPTH", &track_angle, 23, on_trackbarCameraAngle, m_Kinect.pNuiSensor);
	
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
		
		m_Kinect.getMatrix(m_Kinect.All, colorImg, depthRaw, depth8bit, newTimeStamp);
		m_obstacle.setCameraAngle(m_Kinect.getAngle());
		if (newTimeStamp <= oldTimeStamp)
			continue;
		oldTimeStamp = newTimeStamp;

		m_obstacle.setCurrentColor(&colorImg);
		
		m_obstacle.SetCurrentRawDepth(&depthRaw);
		m_obstacle.run(&depth8bit);
		if (!m_Kinect.replay)
			m_obstacle.findHole(m_Kinect.pNuiSensor);
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


	HANDLE handle = GetStdHandle(STD_INPUT_HANDLE);
	DWORD events;
	std::string faceName;

	while (waitKey(1) != ESCAPE_KEY) {
		if (finished)
			return;
		m_Kinect.getColor(colorImg, newTimeStamp);
		if (newTimeStamp <= oldTimeStamp)
			continue;
		oldTimeStamp = newTimeStamp;

		INPUT_RECORD buffer;
		PeekConsoleInput(handle, &buffer, 1, &events);
		if (events > 0 && !m_Kinect.replay && !m_Kinect.recording)
		{
			ReadConsoleInput(handle, &buffer, 1, &events);
			switch (buffer.Event.KeyEvent.wVirtualKeyCode)
			{
			case 0x30:
				while (true)
				{
					std::cout << "Please enter person's name to be added: ";
					std::getline(std::cin, faceName);
					if ((strcmp(faceName.c_str(), "") != 0) &&
						(strcmp(faceName.c_str(), "0") != 0) &&
						(strcmp(faceName.c_str(), "1") != 0))
						break;
				}
				m_face.isAddNewFace = true;
				m_face.isUpdated = true;
				break;

			case 0x31:
				m_face.isAddNewFace = false;
				break;
			}
		}

		if (m_face.isAddNewFace)
			m_face.addNewFace(colorImg, faceName);
		else
			m_face.runFaceRecognizer(&colorImg);

		cv::imshow("FACE DETECTION", colorImg);
	}
}

void Multithreading::SignDetectionThread_Process()
{
	cv::Mat colorImg;
	uint64_t oldTimeStamp = 0, newTimeStamp = 0;

	while (waitKey(1) != ESCAPE_KEY) {
		if (finished)
			return;

		m_Kinect.getColor(colorImg, newTimeStamp);
		if (newTimeStamp <= oldTimeStamp)
			continue;
		oldTimeStamp = newTimeStamp;
		m_sign.setFrameSize(colorImg.cols, colorImg.rows);
		m_sign.runRecognizer(colorImg);
		//cv::imshow("SIGN DETECTION", colorImg);
	}
}

void Multithreading::StairDetectionThread_Process()
{
	cv::Mat colorImg, depth8bit, depthRaw;
	uint64_t oldTimeStamp = 0, newTimeStamp = 0;
	std::vector<cv::Point> stairConvexHull;
	int previousFound = 0;
	int foundThreshold = 3;
	while (waitKey(1) != ESCAPE_KEY) {
		if (finished)
			return;

		m_Kinect.getMatrix(m_Kinect.ColorDepth8bit, colorImg, Mat(), depth8bit, newTimeStamp);
		if (newTimeStamp <= oldTimeStamp)
			continue;
		oldTimeStamp = newTimeStamp;

		m_stairs.Run(colorImg, depth8bit, stairConvexHull);
		if (!stairConvexHull.empty()) {
			if (previousFound > foundThreshold) {
				TextToSpeech::pushBack(string("Stairs Found"));
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

