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
			m_depthImage.create(m_depthFrame.getHeight(), m_depthFrame.getWidth(), CV_16UC1);
			m_depthImage.data = (uchar*)m_depthFrame.getData();

			this->m_depthTimeStamp = m_depthFrame.getTimestamp() >> 16;
			std::cout << "Depth Timestamp: " << this->m_depthTimeStamp << std::endl;
			depthCaptured = true;
			break;
		case C_COLOR_STREAM:
			m_color.readFrame(&m_colorFrame);
			m_colorImage.create(m_colorFrame.getHeight(), m_colorFrame.getWidth(), CV_8UC3);
			m_colorImage.data = (uchar*)m_colorFrame.getData();

			cv::cvtColor(m_colorImage, m_colorImage, CV_BGR2RGB);

			this->m_colorTimeStamp = m_colorFrame.getTimestamp() >> 16;
			std::cout << "Color Timestamp: " << m_colorTimeStamp << std::endl;
			colorCaptured = true;
			break;
		default:
			break;
		}
	}
}

cv::Mat OpenCVKinect::getColor()
{
	return m_colorImage;
}

cv::Mat OpenCVKinect::getDepthRaw()
{
	return m_depthImage;
}

cv::Mat OpenCVKinect::getDepth8bit(cv::Mat &result)
{
	double min, max;
	minMaxLoc(m_depthImage, &min, &max);
	m_depthImage.convertTo(result, CV_8U, 255.0/max);
	return result;
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
