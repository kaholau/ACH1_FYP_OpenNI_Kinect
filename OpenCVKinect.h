#pragma once

#include <OpenNI.h>
#include <cv.h>
#include <highgui.h>
#include <vector>

#define C_DEPTH_STREAM 0
#define C_COLOR_STREAM 1

#define C_NUM_STREAMS 2

#define C_STREAM_TIMEOUT 2000
class OpenCVKinect
{
	openni::Status m_status;
	openni::Device m_device;
	openni::VideoStream m_depth, m_color, **m_streams;
	openni::VideoFrameRef m_depthFrame, m_colorFrame;
	int m_currentStream;
	uint64_t m_depthTimeStamp, m_colorTimeStamp;
	cv::Mat m_depthImage, m_colorImage;
	bool m_alignedStreamStatus, m_colorStreamStatus, m_depthStreamStatus;
public:
	OpenCVKinect(void);
	~OpenCVKinect(void);

	bool init();

	void updateData();
	cv::Mat getDepthRaw();
	cv::Mat getDepth8bit(cv::Mat &result);
	cv::Mat getColor();

};