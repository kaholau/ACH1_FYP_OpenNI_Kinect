#pragma once
#include <Windows.h>
#include <NuiApi.h>
#include <stdlib.h>

#include <OpenNI.h>
#include <cv.h>
#include <highgui.h>
#include <vector>
#include <mutex>

#include <iostream>
#include <fstream>
#include <queue>

#define C_DEPTH_STREAM 0
#define C_COLOR_STREAM 1

#define C_NUM_STREAMS 2

#define C_STREAM_TIMEOUT 5000
class OpenCVKinect
{

	std::mutex color_mutex, depth_mutex;
	openni::Status m_status;
	openni::Device m_device;
	openni::VideoStream m_depth, m_color, **m_streams;
	openni::VideoFrameRef m_depthFrame, m_colorFrame;
	int m_currentStream;
	uint64_t m_depthTimeStamp, m_colorTimeStamp;
	cv::Mat m_depthImage, m_colorImage;
	bool m_alignedStreamStatus, m_colorStreamStatus, m_depthStreamStatus;

	std::string timestamp = "232062183209";
	std::string path = "";
	std::queue<int> angles;
public:
	static enum MatFlag
	{
		None = 0,
		ColorOnly = 1,
		DepthRawOnly = 2,
		ColorDepthRaw = 3,
		Depth8bit = 4,
		ColorDepth8bit = 5,
		DepthRawDepth8bit = 6,
		All = 7
	};
	OpenCVKinect(void);
	~OpenCVKinect(void);

	INuiSensor *pNuiSensor;
	openni::Recorder m_recorder;
	std::ofstream file;
	bool recording = false;

	bool init();
	
	void updateData();
	void getDepthRaw(cv::Mat &depthRaw, uint64_t &depthTimeStamp);
	void getDepth8bit(cv::Mat &depth8bit, uint64_t &depthTimeStamp);
	void getColor(cv::Mat &colorMat, uint64_t &colorTimeStamp);
	void getMatrix(MatFlag type, cv::Mat &color, cv::Mat &depthRaw, cv::Mat &depth8bit, uint64_t &timestamp);
	LONG getAngle();
};