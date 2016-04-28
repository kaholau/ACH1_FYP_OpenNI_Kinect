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
#define TEST false
//#define FOR_REPORT
//#define DISPLAY_HIST
//#define DISPLAY_HEIGHT	
//#define DISPLAY_DISTANCE
//#define DISPLAY_HULL
//#define DISPLAY_ARROW
#define DISPLAY_DIR_LINE

/*Ground detection*/
#define  GROUND_HEIGHT 460		//mm
#define  INIT_CAMERA_ANGLE -6//degree
#define SQUARE_PLANE_EDGE 8 
#define SQUARE_PLANE_AREA (SQUARE_PLANE_EDGE * SQUARE_PLANE_EDGE)
#define MAX_TH_NORMAL 0.5	//angle of surface normal vector 
#define MIN_TH_NORMAL 0.01	//angle of surface normal vector 
#define DILATION_SIZE 2 //if 1, the size is 2*dilation_size1+1
/*hole detection*/
#define GROUND_MIN_HIEGHT -200
#define HOLE_DETECED_ONE_CONFIRM_COUNT 10
#define HOLE_DETECED_CONFIRM_COUNT 7
#define HOLE_DETECTED_SPEECH "hole detected"
#define NO_HOLE_DETECTED_SPEECH "no hole"
#define MOTOR_LOOK_DOWN	-21
/*Obstacle detection*/
#define HISTOGRAM_SIZE 256 // bin size = 2^pixelDepth / histSize
#define  TOO_FAR_PIXEL_VALUE 60		//pixel value
#define OBSTACLE_SIZE_IGNORE SQUARE_PLANE_EDGE*2  
#define MAX_NUM_LOCALMINMA 15
/*Path direction*/
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
	int mUserHeight;
	Mat outputDepth8bit; //this is the depth used for imshow in multithreading.cpp
	Mat currentDepth16bit;
	Mat currentDepth8bit;
	/*ground detection*/
	Mat GroundMat;
	Mat GroundBoolMat;
	/*detect hole*/
	bool angleSetToLookDown = false;
	bool holeDetectedInOneFrame = false;
	bool holeDetected = false;
	bool restoreAngle = false;
	int holeDetectedCountInOneFrame = 0;
	int holeDetectedCount = 0;
	/*obstacle detection*/
	Mat* pThreasholdImageList;
	Mat obstacleMask;
	Mat MaskLayer1;
	Mat MaskLayer2;
	vector<Object> ObstacleList;
	/*Path direction*/
	int currentPathDir; // 
	int currentPathDirCol; //col value of the center of mass of ground
	int right_line = 82; 
	int right_middle_line = 127;
	int left_middle_line = 172;
	int left_line = 217;
	Serial serial;

private:
	//Height
	double GetPointAngle(const int pointY);
	double GetHeight(const int pointY, const int depth);
		
	//Square plane filter
	void GroundMaskCreate();
	void GroundBoolMatFill(Point& location, Vec3f& vector);
	void GroundBoolMatToGroundMat();
	void GroundArrowDraw(Mat& img, Vec3f& vector, Point& start);
	void GroundArrowDrawOnWhitePaper(Mat& img, Vec3f& vector, Point& start);
	float GroundDirection(Vec3f& vector);
	
	//Histogram Segmentation for obstacle detection
	Mat HistogramCal();
	vector<int> HistogramLocalMinima(Mat& hist);
	int getColorIndex(int pixelValue, int index[], int indexSize);
	void Segmentation();
	void SegementLabel(vector<int> &localMin);
	void obstacleDetect(Mat& img, Mat& output);
	void createObstacle(vector<Point> contour, Point center);

	//find path
	int findPathByMassCenter();

	void output();

public:
	const int initCameraAngle = INIT_CAMERA_ANGLE;
	ObstacleDetection(int userHeight);
	~ObstacleDetection();
	void init(Size depthResolution);
	void run(Mat* depth8bit, Mat* depth16bit, int angle);
	void getOutputDepthImg(Mat* depth);
	//find hole
	int findHole();
	
	bool test = TEST;
};


