// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once
#include <Windows.h>
#include <NuiApi.h>
#include <stdlib.h>

#include <ctime>
#include <stdio.h>
#include "stdint.h"
#include <stdlib.h>
#include <algorithm>
#include <tchar.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp> //imread

#include "TextToSpeech.h"
#include "StairDetection.h"
#include "SerialClass.h"

using namespace cv;
//#define FOR_REPORT
//#define TEST_SEGMENTATION
//#define DISPLAY_HIST
//#define record_hist
//#define DISPLAY_HEIGHT	
//#define DISPLAY_DISTANCE
//#define DISPLAY_HULL
//#define record_noGround
//#define DISPLAY_ARROW
//#define COLLECT_TAN_LIST
#define DISPLAY_DIR_LINE

#define  Valid_Distance 1		//grayscale pixel value
#define  TooFarDistance 60
#define  Ground_height 460		//30cm
#define  INIT_CAMERA_ANGLE -6
//calHistogram
#define HistSize 256 // bin size = 2^pixelDepth / histSize

//remove square plane in ground detection
#define planeEdgeForPlaneRemove 8 
#define planeAreaForPlaneRemove (planeEdgeForPlaneRemove * planeEdgeForPlaneRemove)
#define maxThreashold_horizontalPlane 0.5	//angle of surface normal vector 
#define minThreashold_horizontalPlane 0.01	//angle of surface normal vector 
//#define humanShoulderLength 70
#define DILATION_SIZE 1 //if 1, the size is 2*dilation_size1+1
//obstacle detection
#define obstacle_size_ignore planeEdgeForPlaneRemove*2  

//find hole
#define GROUND_MIN_HIEGHT -200
#define HOLE_DETECED_ONE_CONFIRM_COUNT 10
#define HOLE_DETECED_CONFIRM_COUNT 7
#define HOLE_DETECTED_SPEECH "hole detected"
#define NO_HOLE_DETECTED_SPEECH "no hole"
#define MOTOR_LOOK_DOWN	-27

//path direction
#define TURN_LEFT			1
#define MIDDLE_LEFT			2
#define MIDDLE_MIDDLE		3
#define MIDDLE_RIGHT		4
#define TURN_RIGHT			5
#define NO_PATH				0

class ObstacleDetection
{
	struct Object
	{
		vector<vector<Point>> contour;
		Point pos;
	};

	Size DepthMatSize;
	int DepthMatRow;
	int DepthMatCol;
	int CameraAngle = 0;
	Mat currentDepth16bit;
	Mat currentDepth8bit;
	Mat GroundMat;
	Mat GroundBoolMat;
	vector<Object> ObstacleList;
	int mUserHeight;
	int currentPathDir; // 
	int currentPathDirCol; //col value of the center of mass of ground

	Serial serial;

private:


	//Height
	double GetPointAngle(const int pointY);
	double GetHeight(const int pointY, const int depth);
	void createObstacle(vector<Point> contour, Point center);
	
	//Square plane filter
	void GroundMaskCreate();
	void GroundBoolMatFill(Point& location, Vec3f& vector);
	void GroundBoolMatToGroundMat();
	void GroundArrowDraw(Mat& img, Vec3f& vector, Point& start);
	void GroundArrowDrawOnWhitePaper(Mat& img, Vec3f& vector, Point& start);
	float GroundDirection(Vec3f& vector);
	
	
	//Histogram Segmentation for obstacle detection
	Mat HistogramCal(Mat& img);
	vector<int> HistogramLocalMinima(Mat& hist);
	int getColorIndex(int pixelValue, int index[], int indexSize);
	void Segmentation(Mat& src);
	void SegementLabel(Mat& src, vector<int> &localMin);
	void obstacleDetect(Mat& img, Mat& output);

	//find path
	int findPathByMassCenter();

	//find hole
	bool angleSetToLookDown = false;
	bool holeDetectedInOneFrame = false;
	bool holeDetected = false;
	bool restoreAngle = false;
	int holeDetectedCountInOneFrame = 0;
	int holeDetectedCount = 0;

public:
	const int initCameraAngle = INIT_CAMERA_ANGLE;

	ObstacleDetection(int userHeight);
	~ObstacleDetection();
	void run(Mat* depth8bit, Mat* depth16bit, int angle);
	void getOutputDepthImg(Mat* depth);
	//find hole
	int findHole();
};


