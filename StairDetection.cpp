#include "StairDetection.h"


StairDetection::StairDetection()
{
}


StairDetection::~StairDetection()
{
}

void StairDetection::Run(cv::InputArray colorImg, cv::InputArray depthImg, std::vector<cv::Point> &stairsConvexHull)
{
	cv::Mat detected_edges, detected_edges_inv;
	std::vector<cv::Point> stairsConvexHull_normal, stairsConvexHull_inverse;
	cv::Mat scaledColor, scaledDepth;
	std::vector<cv::Vec4i> allLines;
	int stairsAngle = -999;
	cv::Point stairMidPoint;
	std::vector<std::vector<int>> angles = std::vector<std::vector<int>>(180, std::vector<int>());
	std::vector<cv::Point> stairPoints, stairsHull;
	std::vector<cv::Point> stairMidLine, stairsMidPoints;
	std::string timestamp = std::to_string(cv::getTickCount());

	cv::resize(colorImg, scaledColor, cv::Size(320, 240));
	cv::resize(depthImg, scaledDepth, cv::Size(320, 240));
	//cv::imshow("Color Stairs", scaledColor);
	cv::imshow("DEPTH Stairs", scaledDepth);
	CannyThreshold(scaledColor, detected_edges);
	ApplyFilter(detected_edges, scaledDepth, 254, 255, CV_THRESH_BINARY);
	Probabilistic_Hough(detected_edges, allLines);

	cv::cvtColor(detected_edges, detected_edges_inv, CV_GRAY2BGR);
	cv::Scalar color(cv::theRNG().uniform(0, 255), cv::theRNG().uniform(0, 255), cv::theRNG().uniform(0, 255));
	for (cv::Vec4i vec : allLines) {
		cv::Point p1(vec[0], vec[1]);
		cv::Point p2(vec[2], vec[3]);
		cv::line(detected_edges_inv, p1, p2, color, 3);
	}
	cv::imshow("asdf", detected_edges_inv);

	SortLinesByAngle(allLines, angles);
	DetermineStairAngle(angles, stairsAngle);

	// stairs angle not found.
	if (stairsAngle == -999)
		return;

	GetStairMidLine(allLines, angles[stairsAngle], stairsAngle, stairMidLine);
	GetStairPoints(allLines, stairMidLine, stairsAngle, stairPoints, stairsMidPoints);

	cv::cvtColor(detected_edges, detected_edges_inv, CV_GRAY2BGR);
	for (int i = 0; i < stairPoints.size() / 2; i+=2) {
		cv::Point p1(stairPoints[i]);
		cv::Point p2(stairPoints[i+1]);
		cv::line(detected_edges_inv, p1, p2, color, 3);
	}
	cv::imshow("qwerty", detected_edges_inv);

	if (!DetermineStairs(scaledDepth, stairMidLine, stairsMidPoints))
		return;

	ExtractStairsHull(stairPoints, stairsConvexHull);
	return;
}

bool StairDetection::DetermineStairs(cv::InputArray depthImg, std::vector<cv::Point> &stairMidLine, std::vector<cv::Point> &stairMidPoints)
{

	// current holds the current found depth.
	int current = -1, above = -1, below = -1;
	int previous = -1, zeroCount = 0;

	const int ZeroConsequtiveLimit = 10;
	const int PreviousDeltaAllowance = 5;
	const int MaxDepth = 150;

	if (cv::waitKey(1) == 's') {
		cv::LineIterator it1(depthImg.getMat(), stairMidLine[0], stairMidLine[1], 8, false);

		std::string timestamp = std::to_string(cv::getTickCount());
		std::cout << "saving " << timestamp << std::endl;

		std::ofstream file;
		file.open(timestamp + ".txt");
		for (int i = 0; i < it1.count / 2.0; i++, ++it1)
		{
			current = (int)depthImg.getMat().at<uchar>(it1.pos());
			file << current << std::endl;
		}

		file << std::endl;
		file.close();
	}

	cv::LineIterator it(depthImg.getMat(), stairMidLine[0], stairMidLine[1], 8, false);
	std::vector<cv::Point>::iterator midpts = stairMidPoints.begin();

	for (int i = 3; i < it.count / 2.0; i++, ++it) 
	{
		// skip if this is not the stair edge.
		// the stair edges are stored in midpts.
		if (i != midpts->y)
			continue;

		cv::Point curPt(it.pos());
		cv::Point abvPt(it.pos().x, it.pos().y + 3);
		cv::Point blwPt(it.pos().x, it.pos().y - 3);

		current = (int)depthImg.getMat().at<uchar>(curPt);
		above = (int)depthImg.getMat().at<uchar>(abvPt);
		below = (int)depthImg.getMat().at<uchar>(blwPt);

		/// old code?
		if (current == 0) {
			++zeroCount;
			/// if too many consecutive zeroes, 
			/// then this image is too corrupted / does not have stairs
			if (zeroCount > ZeroConsequtiveLimit)
				return false;

			continue;
		}
		zeroCount = 0;

		/// Let current = depth at stairEdge at y;
		///     below = depth at stairEdge at y - 3;
		///     above = depth at stairEdge at y - 3;
		/// then above similar to current AND 
		///      below < current;
		/// else not stairs;

		// if absolute difference between above and current is < 3;
		// difference between current's depth must more than 9 units.
		if (((above - current) < 3) && (current - below) > 9)
			continue;
		else
			return false;

		// update to next stair edge.
		++midpts;
	}


	//cv::LineIterator it(depthImg.getMat(), stairMidLine[0], stairMidLine[1], 8, false);
	//// until first half of the image.
	//for (int i = 0; i < it.count / 2.0; i++, ++it)
	//{
	//	current = (int)depthImg.getMat().at<uchar>(it.pos());

	//	if (current == 0) {
	//		++zeroCount;
	//		/// if too many consecutive zeroes, 
	//		/// then this image is too corrupted / does not have stairs
	//		if (zeroCount > ZeroConsequtiveLimit)
	//			return false;

	//		continue;
	//	}
	//	zeroCount = 0;

	//	/// If depth is very large, then user is too close to object.
	//	/// impossible to be stairs.
	//	if (current > MaxDepth)
	//		return false;

	//	/// Stairs should have ascending depth value;
	//	/// However, on angled view stairs, the depth value can occasional drop a bit.
	//	/// Else it's not stairs at all.
	//	if (current > previous)
	//		previous = current;
	//	else if (current > (previous - PreviousDeltaAllowance))
	//		continue;
	//	else
	//		return false;
	//}

	return true;
}

void StairDetection::GetIntersectHull(std::vector<cv::Point> &stairsConvexHull_normal, std::vector<cv::Point> &stairsConvexHull_inverse, std::vector<cv::Point> &intersectConvexHull)
{
	if (stairsConvexHull_normal.empty() || stairsConvexHull_inverse.empty())
		return;

	for (cv::Point p : stairsConvexHull_normal) {
		if (pointPolygonTest(stairsConvexHull_inverse, p, false) >= 0) {
			intersectConvexHull.push_back(p);
		}
	}

	for (cv::Point p : stairsConvexHull_inverse) {
		if (pointPolygonTest(stairsConvexHull_normal, p, false) >= 0) {
			intersectConvexHull.push_back(p);
		}
	}
}

int StairDetection::AngleBetween(const int x1, const int y1, const int x2, const int y2) {
	double xDiff = x2 - x1;
	double yDiff = y2 - y1;
	int angle = floor(atan2(yDiff, xDiff) * 180 / CV_PI);
	if (angle < 0) {
		angle += 180;
	}

	return angle;
}

void StairDetection::CannyThreshold(cv::InputArray image, cv::Mat &edges) {
	if (image.channels() > 1) {
		/// Convert the image to grayscale
		cvtColor(image, edges, CV_RGB2GRAY);

		/// Reduce noise with a kernel 3x3
		blur(edges, edges, cv::Size(3, 3));
	}
	else {
		/// src is already a grayscale image.
		edges = image.getMat().clone();
	}

	/// Canny detector
	Canny(edges, edges, cannyLowThreshold, cannyLowThreshold*cannyRatio, cannyKernelSize);
}

void StairDetection::ApplyFilter(cv::Mat &src, cv::InputArray &filter, double thresh, double maxval, int type) {
	cv::Mat threshold, temp;
	cv::blur(filter, temp, cv::Size(3, 3));
	cv::threshold(temp, threshold, 0, 255, CV_THRESH_BINARY);

	int erosion_size = 1;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));

	/// Apply the erosion operation
	dilate(threshold, temp, element);
	cv::bitwise_and(src.clone(), temp, src);
}

void StairDetection::Probabilistic_Hough(cv::InputArray src, cv::OutputArray output) {
	HoughLinesP(src, output, 2, CV_PI / 720.0 * 1.0, min_Houghthreshold + houghThreshold, min_HoughLinelength, min_HoughLinegap);
}

/// groups the angles into one similar angle.
int StairDetection::GroupAngles(int angle)
{
	/// ignore angles between this range.
	/// because stairs shouldn't be perpendicular to the user.
	if (angle > 33 & angle < 147)
		return -1;

	/// shift the angle into the middle angle.
	int binSize = 10;

	if (angle % binSize <= binSize / 2.0)
		angle -= angle % binSize;
	else
		angle += (-angle % binSize);

	if (angle > 180)
		angle = 0;

	return angle;
}

/// Determine the angles in the list of lines in allLines
/// and sorts the lines of the same angle into the same vector
void StairDetection::SortLinesByAngle(std::vector<cv::Vec4i> &allLines, std::vector<std::vector<int>> &angles)
{

	for (size_t i = 0; i < allLines.size(); i++) {
		cv::Vec4i l = allLines.at(i);
		int angle = AngleBetween(l[0], l[1], l[2], l[3]);

		angle = GroupAngles(angle);
		if (angle == -1)
			continue;

		angles[angle].push_back(i);

	}
}

/// Find the angle with the most lines to be the angle of the stairs.
void StairDetection::DetermineStairAngle(std::vector<std::vector<int>> &angles, int &stairsAngle)
{
	int max = 0;
	for (int angle = 0; angle < 180; ++angle) {
		std::vector<int> &vec = angles.at(angle);

		if (!vec.empty() && vec.size() > 3 && vec.size() > max) {
			stairsAngle = angle;
			max = vec.size();
		}
	}
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool StairDetection::intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point r)
{
	cv::Point2f x = o2 - o1;
	cv::Point2f d1 = p1 - o1;
	cv::Point2f d2 = p2 - o2;

	float cross = d1.x*d2.y - d1.y*d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;

	if (r.x < 0 || r.x > 320)
		return false;
	if (r.y < 0 || r.y > 240)
		return false;

	return true;
}

/// From confident stair points,
/// find the best fit line that represents stairs.
void StairDetection::GetStairMidLine(std::vector<cv::Vec4i> &allLines, std::vector<int> &stairIndexes, int &stairsAngle, std::vector<cv::Point> &stairMidLine)
{
	std::vector<cv::Point2f> points;
	cv::Vec4f temp;
	for (int index : stairIndexes) {
		cv::Vec4i &l = allLines[index];

		cv::Point p1(l[0], l[1]);
		cv::Point p2(l[2], l[3]);

		points.push_back(p1);
		points.push_back(p2);
	}

	if (!points.empty())
		cv::fitLine(points, temp, CV_DIST_L2, 0, 0.01, 0.01);

	cv::Point pt1, pt2;
	double theta = stairsAngle * CV_PI / 180;
	double a = cos(theta), b = sin(theta);

	pt1.x = cvRound(temp[2] + 320 * -b);
	pt1.y = cvRound(temp[3] + 240 * a);
	pt2.x = cvRound(temp[2] - 320 * -b);
	pt2.y = cvRound(temp[3] - 240 * a);

	if (pt1.x < 0)
		pt1.x = 0;
	if (pt2.x < 0)
		pt2.x = 0;
	if (pt1.y < 0)
		pt1.y = 0;
	if (pt2.y < 0)
		pt2.y = 0;
	if (pt1.x >= 320)
		pt1.x = 319;
	if (pt2.x >= 320)
		pt2.x = 319;
	if (pt1.y >= 240)
		pt1.y = 239;
	if (pt2.y >= 240)
		pt2.y = 239;

	/// Ensure the lower point is first.
	if (pt1.y > pt2.y) {
		stairMidLine.push_back(pt1);
		stairMidLine.push_back(pt2);
	}
	else {
		stairMidLine.push_back(pt2);
		stairMidLine.push_back(pt1);
	}


}

/// Using the found best fit line that represents the stairs,
/// find all lines that intersect with the fit line
/// all lines that intersect belong to the stairs
void StairDetection::GetStairPoints(std::vector<cv::Vec4i> &allLines, std::vector<cv::Point> &stairMidLine, int &stairsAngle, std::vector<cv::Point> &stairPoints, std::vector<cv::Point> &stairMidPoints)
{
	cv::Point pt1 = stairMidLine[0];
	cv::Point pt2 = stairMidLine[1];

	for (cv::Vec4i vec : allLines) {
		cv::Point l1(vec[0], vec[1]);
		cv::Point l2(vec[2], vec[3]);
		cv::Point r;
		int theta = AngleBetween(l1.x, l1.y, l2.x, l2.y);

		// if the line is within +- 10 degs, and it intersects the midline, 
		//   then it belongs to the set of lines that represents the stairs.
		if (abs(stairsAngle - theta) <= 10) {
			if (intersection(l1, l2, pt1, pt2, r)) {
				stairPoints.push_back(l1);
				stairPoints.push_back(l2);
				stairMidPoints.push_back(r);
			}
		}
	}
}

void StairDetection::ExtractStairsHull(std::vector<cv::Point> &stairPoints, std::vector<cv::Point> &stairsHull)
{
	if (!stairPoints.empty())
		cv::convexHull(cv::Mat(stairPoints), stairsHull);
}

void StairDetection::drawStairs(std::string windowName, cv::InputArray colorImg, std::vector<cv::Point> &stairConvexHull)
{
	if (!stairConvexHull.empty()) {
		cv::Mat temp;
		cv::resize(colorImg, temp, cv::Size(320, 240));
		std::vector<std::vector<cv::Point> > hull(1);
		cv::Scalar color = cv::Scalar(cv::theRNG().uniform(0, 255), cv::theRNG().uniform(0, 255), cv::theRNG().uniform(0, 255));
		hull.push_back(stairConvexHull);
		for (int i = 0; i<hull.size(); ++i) {
			drawContours(temp, hull, i, color, 3, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
		}
		cv::imshow(windowName, temp);
		//cv::imwrite(windowName + ".png", temp);
	}
}
