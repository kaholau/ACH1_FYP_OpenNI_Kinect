// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include <Windows.h>
#include <ctime>
#include <stdio.h>
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

#define SAMPLE_IMG_PATH "C:/Users/HOHO/Pictures/sample/result/"
#define MYFILE_PATH  "C:/Users/HOHO/Pictures/sample/output.txt"
#define SAMPLE_PATH "C:/Users/HOHO/Pictures/sample/depth9.bmp"
//#define SAMPLE_PATH "C:/Users/HOHO/Pictures/sample/Pictures/atrium/depth7.bmp"

#define  Valid_Distance 1		//grayscale pixel value
#define TooFarDistance 60
#define  Ground_height 175		//4cm
#define  TooLessGroundPercentage 0.03 
//calHistogram
#define HistSize 256 // bin size = 2^pixelDepth / histSize

//remove plane
//#define planeEdgeForPlaneRemove 4 //e.g 20X20 square for 320x240
#define planeEdgeForPlaneRemove 8 //e.g 20X20 square for 640x480
#define planeAreaForPlaneRemove (planeEdgeForPlaneRemove * planeEdgeForPlaneRemove)
#define maxThreashold_horizontalPlane 0.33
#define minThreashold_horizontalPlane -0.33
#define maxThreashold_VerticalPlane   1.6
#define minThreashold_VerticalPlane   0.8

//obstacle detection
//#define obstacle_size_ignore 15  //320x240
#define obstacle_size_ignore 27  //640x480 

//path advice	
#define FirstNumOfBin	20				//affect the num of bin in the first histogram calculation
#define SecondHistogramRangeColDivisor 5	//affect the width of rect in the second histogram calculation
#define SecondNumOfBin 7				//affect the num of bin in the second histogram calculation
#define SecondBinHeightDivisor 10		//affect the heigth of the rect in the second histogram calculation

enum ObjectType { UNDEFINED, GROUND, RIGHT_PLANE, LEFT_PLANE, ALL_OBSTACLE, OBSTACLE };
enum Position {left,center,right};

class ObstacleDetection
{
	struct Object
	{
		Mat img;
		vector<vector<Point>> contour;
		double area;
		ObjectType type;
		Point pos;
	};

	std::ofstream myfile;
	Mat currentRawDepth;
	Mat currentDepth;
	Mat currentColor;
	Object Ground;
	Mat GroundBoolMat;
	vector<Object> GroundList;
	vector<Object> ObstacleList;
	int mUserHeight;
	std::queue<float> diplay;
	string currentPath;
	
private:
	//Height
	double GetPointAngle(const int pointY);
	double GetHeight(const int pointY, const int depth);
	void createObstacle(vector<Point> contour, ObjectType type, Point center);
	void createObstacle(vector<Point> contour, ObjectType type, Point center, double area);
	void createObstacle(vector<Point> contour, ObjectType type, double area);
	
	//Plane filter
	int Rand();
	Point RandPoint(Point start);
	Vec3f RandVector(Vec3f &vec1, Vec3f &vec2);
	void GroundMaskFill(Mat& img, Mat& boolMat, Point& location, Vec3f& vector);
	void GroundMaskBool();
	void GroundBoolMatUnitFill(Mat& fill, Point& pt);
	void GroundArrowDraw(Mat& img, Vec3f& vector, Point& start);
	void GroundArrowDrawOnWhitePaper(Mat& img, Vec3f& vector, Point& start);
	void GroundMaskUnitFill(Mat& fill, Point& pt);
	void createPlaneObject(Mat& src, Mat& img, ObjectType type);
	void GroundMaskCreateRegular(Mat &img);
	void GroundMaskCreateRandom(Mat &img);
	void GroundDefault(Mat& img);
	void GroundEroAndDilate();
	void GroundRefine(Mat& boolMat, Mat& img);

	//Histogram Segmentation
	Mat HistogramCal(Mat& img);
	vector<int> HistogramLocalMinima(Mat& hist);
	int getColorIndex(int pixelValue, int index[], int indexSize);
	void Segmentation(Mat& src);
	void SegementLabel(Mat& src, vector<int> &localMin);
	void obstacleDetect(Mat& img, Mat& output);

	//find path
	string findPath();
	void Enhance1DMax(Mat *pImg);

public:

	int erosion_elem = 0;
	int erosion_size = 0;
	int dilation_elem = 0;
	int dilation_size = 0;

	int CameraAngle;
	ObstacleDetection(int userHeight);
	~ObstacleDetection();
	void run(Mat* src);
	void getOutputDepthImg(Mat* depth);
	void getOutputColorImg(Mat *color);
	int getCameraAngle();
	void setCurrentColor(Mat* src);
	void setCameraAngle(int degree);
	void SetCurrentRawDepth(Mat* rawDepth);
	void test();

};


