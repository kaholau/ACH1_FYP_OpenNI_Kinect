#pragma once

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <queue>

class StairDetection
{
private:
	static const int cannyLowThreshold = 20;
	static const int max_cannyLowThreshold = 100;
	static const int cannyRatio = 3;
	static const int cannyKernelSize = 3;

	static const int houghThreshold = 60;
	static const int min_Houghthreshold = 50;
	static const int max_HoughThreshold = 150;
	static const int min_HoughLinelength = 60;
	static const int min_HoughLinegap = 5;

	void CannyThreshold(cv::InputArray image, cv::Mat &edges);
	void Probabilistic_Hough(cv::InputArray src, cv::OutputArray output);
	void ApplyFilter(cv::Mat &src, cv::InputArray &filter, double thres, double maxval, int type);
	void SortLinesByAngle(std::vector<cv::Vec4i> &p_lines, std::vector<std::vector<int>> &angles);
	void DetermineStairAngle(std::vector<std::vector<int>> &angles, int &stairsAngle);
	void GetStairMidLine(std::vector<cv::Vec4i> &allLines, std::vector<int> &stairIndexes, int &stairsAngle, std::vector<cv::Point> &stairMidLine);
	void GetStairPoints(std::vector<cv::Vec4i> &allLines, std::vector<cv::Point> &stairMidLine, std::vector<cv::Point> &stairPoints);
	void ExtractStairsHull(std::vector<cv::Point> &stairPoints, std::vector<cv::Point> &stairsHull);
	void GetIntersectHull(std::vector<cv::Point> &stairsConvexHull_normal, std::vector<cv::Point> &stairsConvexHull_inverse, std::vector<cv::Point> &intersectConvexHull);
	bool DetermineStairs(cv::InputArray depthImg, std::vector<cv::Point> &stairMidLine, std::vector<cv::Point> &stairPoints);

	int GroupAngles(int angle);
	int AngleBetween(const int x1, const int y1, const int x2, const int y2);
	bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point r);

public:
	StairDetection();
	~StairDetection();
	static void drawStairs(std::string windowName, cv::InputArray colorImg, std::vector<cv::Point> &stairConvexHull);
	void Run(cv::InputArray colorImg, cv::InputArray depthRawImg, std::vector<cv::Point> &stairsConvexHull);
};

