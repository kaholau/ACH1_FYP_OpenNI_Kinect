#include "OpenCVKinect.h"

OpenCVKinect::OpenCVKinect(void)
{
	m_depthTimeStamp = 0;
	m_colorTimeStamp = 0;

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
	m_status = m_device.open(deviceURI);
	if(m_status != openni::STATUS_OK)
	{
		std::cout << "OpenCVKinect: Device open failseed: " << std::endl;
		std::cout << openni::OpenNI::getExtendedError() << std::endl;
		openni::OpenNI::shutdown();
		return false;
	}

	// create a depth object
	m_status = m_depth.create(m_device, openni::SENSOR_DEPTH);
	if(m_status == openni::STATUS_OK)
	{
		m_status = m_depth.start();
		if(m_status != openni::STATUS_OK)
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
	if(m_status == openni::STATUS_OK)
	{
		m_status = m_color.start();
		if(m_status != openni::STATUS_OK)
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
	m_device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);


	if(!m_depth.isValid() && !m_color.isValid())
	{
		std::cout << "OpenCVKinect: No valid streams. Exiting" << std::endl;
		openni::OpenNI::shutdown();
		return false;
	}

	this->m_streams = new openni::VideoStream*[C_NUM_STREAMS];
	m_streams[C_DEPTH_STREAM] = &m_depth;
	m_streams[C_COLOR_STREAM] = &m_color;

	return true;
}

void OpenCVKinect::updateData()
{
	bool depthCaptured = false, colorCaptured = false;
	uint64_t newtime;

	while( !depthCaptured || !colorCaptured || m_depthTimeStamp != m_colorTimeStamp)
	{
		m_status = openni::OpenNI::waitForAnyStream(m_streams, C_NUM_STREAMS, &m_currentStream, C_STREAM_TIMEOUT);
		if(m_status != openni::STATUS_OK)
		{
			std::cout << "OpenCVKinect: Unable to wait for streams. Exiting" << std::endl;
			exit(EXIT_FAILURE);
		}

		switch(m_currentStream)
		{
		case C_DEPTH_STREAM:
			m_depth.readFrame(&m_depthFrame);
			newtime = m_depthFrame.getTimestamp() >> 16;

			if (newtime <= this->m_depthTimeStamp)
				continue;

			depth_mutex.lock();
			this->m_depthTimeStamp = newtime;
			m_depthImage.create(m_depthFrame.getHeight(), m_depthFrame.getWidth(), CV_16UC1);
			m_depthImage.data = (uchar*)m_depthFrame.getData();

			std::cout << "Depth Timestamp: " << this->m_depthTimeStamp << std::endl;
			depthCaptured = true;
			depth_mutex.unlock();

			break;
		case C_COLOR_STREAM:
			m_color.readFrame(&m_colorFrame);
			newtime = m_colorFrame.getTimestamp() >> 16;

			if (newtime <= this->m_colorTimeStamp)
				continue;

			color_mutex.lock();
			this->m_colorTimeStamp = newtime;
			m_colorImage.create(m_colorFrame.getHeight(), m_colorFrame.getWidth(), CV_8UC3);
			m_colorImage.data = (uchar*)m_colorFrame.getData();

			cv::cvtColor(m_colorImage, m_colorImage, CV_BGR2RGB);

			std::cout << "Color Timestamp: " << m_colorTimeStamp << std::endl;
			colorCaptured = true;
			color_mutex.unlock();

			break;
		default:
			break;
		}
	}
}

void OpenCVKinect::getColor(cv::Mat &colorMat, uint64_t &colorTimeStamp)
{
	color_mutex.lock();
	colorMat = m_colorImage.clone();
	colorTimeStamp = m_colorTimeStamp;
	color_mutex.unlock();
}

void OpenCVKinect::getDepthRaw(cv::Mat &depthRaw, uint64_t &depthTimeStamp)
{
	depth_mutex.lock();
	depthRaw = m_depthImage.clone();
	depthTimeStamp = m_depthTimeStamp;
	depth_mutex.unlock();
}

void OpenCVKinect::getDepth8bit(cv::Mat &depth8bit, uint64_t &depthTimeStamp)
{
	depth_mutex.lock();
	double min, max;
	cv::minMaxLoc(m_depthImage, &min, &max);
	m_depthImage.convertTo(depth8bit, CV_8U, 255.0 / max);
	depth_mutex.unlock();
}

void OpenCVKinect::getMatrix(MatFlag type, cv::Mat &colorMat, cv::Mat &depthRawMat, cv::Mat &depth8bitMat, uint64_t &timestamp)
{
	bool color = type & 1;
	bool depthRaw = type & 2;
	bool depth8bit = type & 4;

	if (color)
		color_mutex.lock();
	if (depthRaw || depth8bit)
		depth_mutex.lock();

	if (color)
		colorMat = m_colorImage.clone();
	if (depthRaw)
		depthRawMat = m_depthImage.clone();
	if (depth8bit) {
		double min, max;
		cv::Mat output, temp;
		depth8bitMat = m_depthImage.clone();
		cv::minMaxLoc(depth8bitMat, &min, &max);
		m_depthImage.convertTo(depth8bitMat, CV_8U, 255.0 / max);
	}

	/// both color/depth timestamps are guaranteed same value
	timestamp = m_colorTimeStamp;

	if (color)
		color_mutex.unlock();
	if (depthRaw || depth8bit)
		depth_mutex.unlock();
}


OpenCVKinect::~OpenCVKinect(void)
{
	this->m_depthFrame.release();
	this->m_colorFrame.release();
	this->m_depth.stop();
	this->m_color.stop();
	openni::OpenNI::shutdown();
	this->m_device.close();
}
