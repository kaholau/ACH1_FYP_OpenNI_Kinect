#pragma once

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
	static const int min_HoughLinegap = 10;

	void CannyThreshold(cv::InputArray image, cv::Mat &edges);
	void Probabilistic_Hough(cv::InputArray src, cv::OutputArray output);
	void ApplyFilter(cv::Mat &src, cv::InputArray &filter, double thres, double maxval, int type);
	void SortLinesByAngle(std::vector<cv::Vec4i> &p_lines, std::vector<std::vector<int>> &angles);
	void DetermineStairAngle(std::vector<std::vector<int>> &angles, int &stairsAngle);
	void GetStairMidLine(std::vector<cv::Vec4i> &allLines, std::vector<int> &stairIndexes, cv::Vec4f stairMidLine);
	void GetStairPoints(std::vector<cv::Vec4i> &allLines, cv::Vec4f stairMidLine, int stairsAngle, std::vector<cv::Point> &stairPoints);
	void ExtractStairsHull(std::vector<cv::Point> &stairPoints, std::vector<cv::Point> &stairsHull);
	void RunInternal(cv::Mat detected_edges, cv::InputArray groundImg, int groundInverseType, std::vector<cv::Point> &stairsConvexHull);
	void GetIntersectHull(std::vector<cv::Point> &stairsConvexHull_normal, std::vector<cv::Point> &stairsConvexHull_inverse, std::vector<cv::Point> &intersectConvexHull);


	int GroupAngles(int angle);
	int AngleBetween(const int x1, const int y1, const int x2, const int y2);
public:
	StairDetection();
	~StairDetection();

	void Run(cv::InputArray colorImg, cv::InputArray depthImg, cv::InputArray groundImg, std::vector<cv::Point> &stairsConvexHull);
};

