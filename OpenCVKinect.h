#pragma once
#include <Windows.h>
#include <NuiApi.h>
#include <stdlib.h>

#include <OpenNI.h>
#include <cv.h>
#include <highgui.h>
#include <vector>
#include <mutex>

#include <fstream>
#include <queue>

#include "include.h"

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
	int frameIndex_color=-1;
	int frameIndex_depth=-1;
	uint64_t m_timeStamp;
	cv::Mat m_depthImage, m_colorImage;
	bool m_alignedStreamStatus, m_colorStreamStatus, m_depthStreamStatus;

	std::string timestamp = "2683639321";
	std::string path = "D:/New folder/";
	std::queue<int> angles;
	std::queue<int> frameIndexFromFile;

	bool initialized = false;
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
	LONG angle = 0;
	bool recording = false;
	bool replay = false;

	bool init();
	
	void updateData();
	void updateDataDepthOnly();
	void getDepthRaw(cv::Mat &depthRaw);
	void getDepth8bit(cv::Mat &depth8bit);
	void getColor(cv::Mat &colorMat);
	void getMatrix(MatFlag type, cv::Mat &color, cv::Mat &depthRaw, cv::Mat &depth8bit);
	uint64_t getTimestamp();
	LONG getAngle();
	void setPlayspeed(int playspeed);

	int getFrameIndexColor();
	int getFrameIndexDepth();
};
