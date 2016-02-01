// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include <Windows.h>
#include <ctime>
#include <stdio.h>
#include <algorithm>
#include <tchar.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp> //imread

using namespace cv;
//#define TEST_SEGMENTATION
//#define DISPLAY_HEIGHT	
//#define DISPLAY_HULL

#define MYFILE_PATH  "C:/Users/HOHO/Pictures/sample/output.txt"
#define SAMPLE_PATH "C:/Users/HOHO/Pictures/sample/depth9.bmp"
//#define SAMPLE_PATH "C:/Users/HOHO/Pictures/sample/Pictures/atrium/depth7.bmp"

#define  undefine_pixel Scalar(255)
#define  Ground_pixel Scalar(0,0,0)
#define  Valid_Distance 61		//grayscale pixel value
#define  Ground_height 250		//4cm
#define  TooLessGroundPercentage 0.1 
//calHistogram
#define HistSize 256 // bin size = 2^pixelDepth / histSize

//remove plane
//#define planeEdgeForPlaneRemove 4 //e.g 20X20 square for 320x240
#define planeEdgeForPlaneRemove 12 //e.g 20X20 square for 640x480
#define planeAreaForPlaneRemove (planeEdgeForPlaneRemove * planeEdgeForPlaneRemove) *4
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
	vector<Object> GroundList;
	vector<Object> ObstacleList;
	static Point pos;
	int mUserHeight;
	int CameraAngle;
	
private:
	//Height
	double GetPointAngle(const int pointY);
	double GetHeight(const int pointY, const int depth);
	void createObstacle(vector<Point> contour, ObjectType type, Point center);
	void createObstacle(vector<Point> contour, ObjectType type, Point center, double area);
	void createObstacle(vector<Point> contour, ObjectType type, double area);
	
	//Plane filter
	void GroundMaskFill(Mat& img, Point& location, Vec3f& vector);
	void GroundArrowDraw(Mat& img, Vec3f& vector, Point& start);
	void GroundMaskUnitFill(Mat& fill, Point& pt);
	void createPlaneObject(Mat& src, Mat& img, ObjectType type);
	void GroundMaskCreate(Mat &img);

	//Histogram Segmentation
	Mat HistogramCal(Mat& img);
	vector<int> HistogramLocalMinima(Mat& hist);
	int getColorIndex(int pixelValue, int index[], int indexSize);
	void Segmentation(Mat& src);
	void SegementLabel(Mat& src, vector<int> &localMin);
	void obstacleDetect(Mat& img, Mat& output);

	//find path
	int findPath();
	void Enhance1DMax(Mat *pImg);

public:
	ObstacleDetection(int userHeight);
	~ObstacleDetection();
	void run(Mat* src);
	void getOutputDepthImg(Mat* depth);
	void getOutputColorImg(Mat *color);
	void setCurrentColor(Mat* src);
	void setCameraAngle(int degree);
	void SetCurrentRawDepth(Mat* rawDepth);
	void test();

};


