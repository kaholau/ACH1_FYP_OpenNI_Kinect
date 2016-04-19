/* ----------------------------------------------------------------------------
** SignRecognizer.h
**
** Date: 2016-01-13
** Description: This is the header file of class 'SignRecognizer'. This class
** is currently for detecting the floor signs in HKUST. More and different
** signs may be recognized later.
** This class requires the use of Microsoft Speech API(SAPI) and Tesseract OCR.
** Please install them and include the path and library first.
**
** Microsoft Speech API (SAPI) 5.3 Reference:
** https://msdn.microsoft.com/en-us/library/ms720161(v=vs.85).aspx
** https://msdn.microsoft.com/en-us/library/ms720163(v=vs.85).aspx
** Microsoft Speech SDK 5.1 Download:
** https://www.microsoft.com/en-us/download/details.aspx?id=10121
**
** Tesseract OCR Information and Download:
** https://github.com/tesseract-ocr
**
**
** Author: Chan Tong Yan, Lau Ka Ho, Joel @ HKUST
** E-mail: yumichanhk2014@gmail.com
**
** --------------------------------------------------------------------------*/
#pragma once

#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <tesseract/baseapi.h>
#include <Windows.h>

#include "TextToSpeech.h"

//#define DURATION_CHECK
//#define SHOW_DEBUG_MESSAGES
//#define SHOW_IMAGE_AND_RESULT
//#define SAVE_IMAGE_AND_RESULT
//#define USE_TRACKBAR

#define WINDOW_NAME_EDGE_MASK	"_Edge_Mask"

#define DEFAULT_FRAME_WIDTH		1280
#define DEFAULT_FRAME_HEIGHT	960
#define MAX_VALUE_8BITS			255
#define MID_VALUE_8BITS			(MAX_VALUE_8BITS/2)
#define ABSOLUTE_WHITE			255
#define ABSOLUTE_BLACK			0

#define CANNY_MAX_THRHD_RATIO		2.3
#define AVG_GRAYSCALE_SCALE			1.35
#define MEDIAN_BLUR_KSIZE			5

#define SIGN_WIDTH_THRESHOLD		100
//#define SAME_CONTOUR_THRESHOLD		0.85
#define SAME_CONTOUR_THRESHOLD		0.5
#define SAME_ELLIPSE_THRESHOLD		0.95

#define MAX_SIGN_WIDTH				(DEFAULT_FRAME_WIDTH / 3)
#define MIN_SIGN_WIDTH				(DEFAULT_FRAME_WIDTH / 16)
#define MAX_SIGN_HEIGHT				(DEFAULT_FRAME_HEIGHT / 3)
#define MIN_SIGN_HEIGHT				(DEFAULT_FRAME_HEIGHT / 12)
#define CONTOUR_AREA_THRESHOLD		1000
#define CONTOUR_AREA_MAX_THRESHOLD	250000

typedef class SignRecognizer
{
public:
	SignRecognizer();
	SignRecognizer(int, int, bool);
	~SignRecognizer();
	void runRecognizer(cv::Mat &frame, std::string fName = "");
	void setFrameSize(int, int);
	void testExample(void);

private:
	static const std::wstring STRING_UPPER_FLOOR[8];
	static const std::wstring STRING_LG1;
	static const std::wstring STRING_LG2;
	static const std::wstring STRING_LG3;
	static const std::wstring STRING_LG4;
	static const std::wstring STRING_LG5;
	static const std::wstring STRING_LG6;
	static const std::wstring STRING_LG7;

	bool isLaterallyInverted;
	double scale;
	int lowThreshold;
	int dilateSize;
	//std::vector<struct DetectionInfo> signsInfo;
	std::string lastDet;

	cv::Mat map_x, map_y;
	cv::Size frameSize;
	tesseract::TessBaseAPI tess;

	bool getResultString(std::string &, std::wstring &);
	void getContoursOfFrame(cv::Mat &, cv::Mat &,
		std::vector<std::vector<cv::Point>> &, std::vector<cv::Vec4i> &);
	cv::Mat* invertLaterally(cv::Mat *sign);
	void updateMap(void);
	void updateMap(cv::Size);
} SignRecognizer;

