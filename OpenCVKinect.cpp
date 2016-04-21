#include "OpenCVKinect.h"

OpenCVKinect::OpenCVKinect(void)
{
	m_timeStamp = -1;

	m_alignedStreamStatus = true;
	m_colorStreamStatus = true;
	m_depthStreamStatus = true;
}

bool OpenCVKinect::init()
{
	m_status = openni::STATUS_OK;
	const char* deviceURI = openni::ANY_DEVICE;
	m_status = openni::OpenNI::initialize();
	std::cout << "After initialization: " << std::endl;
	std::cout << openni::OpenNI::getExtendedError() << std::endl;

	// open the device
	if (replay)
	{
		m_status = m_device.open((path + timestamp + ".oni").c_str());

		std::string line;
		std::ifstream file((path + timestamp + ".txt").c_str());
		while (std::getline(file, line)) {
			int angle = stoi(line);
			angles.push(angle);
		}
	}
	else {
		m_status = m_device.open(deviceURI);
	}

	if (m_status != openni::STATUS_OK)
	{
		std::cout << "OpenCVKinect: Device open failseed: " << std::endl;
		std::cout << openni::OpenNI::getExtendedError() << std::endl;
		openni::OpenNI::shutdown();
		return false;
	}

	// create a depth object
	m_status = m_depth.create(m_device, openni::SENSOR_DEPTH);
	const openni::SensorInfo* sinfo = m_device.getSensorInfo(openni::SENSOR_DEPTH);
	const openni::Array<openni::VideoMode> &videoModes = sinfo->getSupportedVideoModes();
	m_depth.setVideoMode(videoModes[1]);
	if (m_status == openni::STATUS_OK)
	{
		m_status = m_depth.start();
		if (m_status != openni::STATUS_OK)
		{
			std::cout << "OpenCVKinect: Couldn't start depth stream: " << std::endl;
			std::cout << openni::OpenNI::getExtendedError() << std::endl;
			m_depth.destroy();
			return false;
		}
	}
	else
	{
		std::cout << "OpenCVKinect: Couldn't find depth stream: " << std::endl;
		std::cout << openni::OpenNI::getExtendedError() << std::endl;
		return false;
	}

	// create a color object
	m_status = m_color.create(m_device, openni::SENSOR_COLOR);
	const openni::SensorInfo* sinfoColor = m_device.getSensorInfo(openni::SENSOR_COLOR);
	const openni::Array<openni::VideoMode> &videoModesColor = sinfoColor->getSupportedVideoModes();
#ifdef COLOUR_FRAME_USE_HIGHEST_RESOLUTION_1280x960
	m_color.setVideoMode(videoModesColor[0]);
#else
	m_color.setVideoMode(videoModesColor[1]);
#endif
	if (m_status == openni::STATUS_OK)
	{
		m_status = m_color.start();
		if (m_status != openni::STATUS_OK)
		{

			std::cout << "OpenCVKinect: Couldn't start color stream: " << std::endl;
			std::cout << openni::OpenNI::getExtendedError() << std::endl;
			m_color.destroy();
			return false;
		}
	}
	else
	{
		std::cout << "OpenCVKinect: Couldn't find color stream: " << std::endl;
		std::cout << openni::OpenNI::getExtendedError() << std::endl;
		return false;
	}

	// Set Depth to Color Registration / Alignment
	m_device.setDepthColorSyncEnabled(true);
	m_device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);


	if (!m_depth.isValid() && !m_color.isValid())
	{
		std::cout << "OpenCVKinect: No valid streams. Exiting" << std::endl;
		openni::OpenNI::shutdown();
		return false;
	}

	this->m_streams = new openni::VideoStream*[C_NUM_STREAMS];
	m_streams[C_DEPTH_STREAM] = &m_depth;
	m_streams[C_COLOR_STREAM] = &m_color;

	if (recording) {
		timestamp = std::to_string(cv::getTickCount());
		file.open((path + timestamp + ".txt").c_str());
		m_recorder.create((path + timestamp + ".oni").c_str());
		m_recorder.attach(*m_streams[C_COLOR_STREAM]);
		m_recorder.attach(*m_streams[C_DEPTH_STREAM]);
	}


	int iSensorCount = 0;
	HRESULT hr = S_OK;
	hr = NuiGetSensorCount(&iSensorCount);

	// Look at each Kinect sensor
	for (int i = 0; i < iSensorCount; ++i)
	{
		// Create the sensor so we can check status, if we can't create it, move on to the next
		hr = NuiCreateSensorByIndex(i, &pNuiSensor);
		if (FAILED(hr))
		{
			continue;
		}

		// Get the status of the sensor, and if connected, then we can initialize it
		hr = pNuiSensor->NuiStatus();
		if (S_OK == hr)
		{
			break;
		}
	}

	initialized = true;

	return true;
}

//void OpenCVKinect::updateData() {
//	if (m_timeStamp == -1) {
//		string colorFile = "image/95_colour.png";
//		string depthFile = "image/95_depthraw.png";
//		Mat temp = imread(colorFile, CV_LOAD_IMAGE_COLOR);
//		//cv::cvtColor(temp, m_colorImage, CV_BGR2RGB);
//		m_depthImage = imread(depthFile, CV_LOAD_IMAGE_ANYDEPTH);
//		m_timeStamp = 1;
//	}
//}

void OpenCVKinect::updateData()
{
	bool depthCaptured = false, colorCaptured = false;
	uint64_t newtime;

	while (!depthCaptured || !colorCaptured)
	{
		m_status = openni::OpenNI::waitForAnyStream(m_streams, C_NUM_STREAMS, &m_currentStream, C_STREAM_TIMEOUT);
		if (replay && m_device.getPlaybackControl()->getSpeed() == -1) {
			return;
		}

		if (m_status != openni::STATUS_OK)
		{
			std::cout << "OpenCVKinect: Unable to wait for streams. Exiting" << std::endl;
			exit(EXIT_FAILURE);
		}

		switch (m_currentStream)
		{
		case C_DEPTH_STREAM:
			m_depth.readFrame(&m_depthFrame);

			if (recording) {
				pNuiSensor->NuiCameraElevationGetAngle(&angle);
				file << angle << std::endl;
			}

			if (replay)
			{
				if (angles.size() > 0)
				{
					angle = angles.front();
					angles.pop();
				}
			}

			depth_mutex.lock();
			this->m_timeStamp = m_depthFrame.getTimestamp() >> 16;;
			m_depthImage.create(m_depthFrame.getHeight(), m_depthFrame.getWidth(), CV_16UC1);
			m_depthImage.data = (uchar*)m_depthFrame.getData();

			//std::cout << "Depth Timestamp: " << this->m_depthFrame.getFrameIndex() << std::endl;

			depthCaptured = true;
			depth_mutex.unlock();

			break;
		case C_COLOR_STREAM:
			m_color.readFrame(&m_colorFrame);

			color_mutex.lock();
			this->m_timeStamp = m_depthFrame.getTimestamp() >> 16;;
			m_colorImage.create(m_colorFrame.getHeight(), m_colorFrame.getWidth(), CV_8UC3);
			m_colorImage.data = (uchar*)m_colorFrame.getData();

			//std::cout << "Color Timestamp: " << this->m_colorFrame.getFrameIndex() << std::endl;
			colorCaptured = true;
			color_mutex.unlock();

			break;
		default:
			break;
		}
	}

	
}

void OpenCVKinect::getColor(cv::Mat &colorMat)
{
	color_mutex.lock();
	cv::cvtColor(m_colorImage, colorMat, CV_BGR2RGB);
	color_mutex.unlock();
}

void OpenCVKinect::getDepthRaw(cv::Mat &depthRaw)
{
	depth_mutex.lock();
	depthRaw = m_depthImage.clone();
	depth_mutex.unlock();
}

void OpenCVKinect::getDepth8bit(cv::Mat &depth8bit)
{
	depth_mutex.lock();
	double min, max;
	cv::minMaxLoc(m_depthImage, &min, &max);
	m_depthImage.convertTo(depth8bit, CV_8U, 255.0 / max);
	depth_mutex.unlock();
}

void OpenCVKinect::getMatrix(MatFlag type, cv::Mat &colorMat, cv::Mat &depthRawMat, cv::Mat &depth8bitMat)
{
	bool color = type & 1;
	bool depthRaw = type & 2;
	bool depth8bit = type & 4;

	if (color)
		color_mutex.lock();
	if (depthRaw || depth8bit)
		depth_mutex.lock();

	if (color){
		cv::cvtColor(m_colorImage, colorMat, CV_BGR2RGB);
	}
	if (depthRaw)
		depthRawMat = m_depthImage.clone();
	if (depth8bit) {
		double min, max;
		cv::Mat output, temp;
		depth8bitMat = m_depthImage.clone();
		cv::minMaxLoc(depth8bitMat, &min, &max);
		m_depthImage.convertTo(depth8bitMat, CV_8U, 255.0 / max);
	}

	if (color)
		color_mutex.unlock();
	if (depthRaw || depth8bit)
		depth_mutex.unlock();
}

uint64_t OpenCVKinect::getTimestamp() {
	return m_timeStamp;
}

LONG OpenCVKinect::getAngle()
{
	if (!replay)
		pNuiSensor->NuiCameraElevationGetAngle(&angle);

	return angle;
}

void OpenCVKinect::setPlayspeed(int playspeed) {
	m_device.getPlaybackControl()->setSpeed(playspeed);
}

OpenCVKinect::~OpenCVKinect(void)
{
	if (!initialized)
		return;

	file.close();
	this->m_recorder.stop();
	this->m_recorder.destroy();
	this->m_depthFrame.release();
	this->m_colorFrame.release();
	this->m_depth.stop();
	this->m_color.stop();
	openni::OpenNI::shutdown();
	this->m_device.close();
}
