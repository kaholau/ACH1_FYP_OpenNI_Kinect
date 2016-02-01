#include "StairDetection.h"


StairDetection::StairDetection()
{
}


StairDetection::~StairDetection()
{
}

void StairDetection::Run(cv::InputArray colorImg, cv::InputArray depthImg, cv::InputArray groundImg, std::vector<cv::Point> &stairsConvexHull)
{
	cv::Mat detected_edges, detected_edges_inv;
	std::vector<cv::Point> stairsConvexHull_normal, stairsConvexHull_inverse;

	CannyThreshold(colorImg, detected_edges);
	ApplyFilter(detected_edges, depthImg, 254, 255, CV_THRESH_BINARY);
	detected_edges_inv = detected_edges.clone();

	RunInternal(detected_edges, groundImg, CV_THRESH_BINARY, stairsConvexHull_normal);
	RunInternal(detected_edges_inv, groundImg, CV_THRESH_BINARY_INV, stairsConvexHull_inverse);

	GetIntersectHull(stairsConvexHull_normal, stairsConvexHull_inverse, stairsConvexHull);
}

void StairDetection::RunInternal(cv::Mat detected_edges, cv::InputArray groundImg, int groundInverseType, std::vector<cv::Point> &stairsConvexHull)
{
	std::vector<cv::Vec4i> allLines;
	int stairsAngle;
	cv::Point stairMidPoint;
	std::vector<std::vector<int>> angles = std::vector<std::vector<int>>(180, std::vector<int>());
	std::vector<cv::Point> stairPoints, stairsHull;
	cv::Vec4f stairMidLine;

	ApplyFilter(detected_edges, groundImg, 0, 255, groundInverseType);
	Probabilistic_Hough(detected_edges, allLines);
	SortLinesByAngle(allLines, angles);
	DetermineStairAngle(angles, stairsAngle);
	GetStairMidLine(allLines, angles[stairsAngle], stairMidLine);
	GetStairPoints(allLines, stairMidLine, stairsAngle, stairPoints);
	ExtractStairsHull(stairPoints, stairsConvexHull);

	cv::Mat temp;
	cv::cvtColor(detected_edges, temp, CV_GRAY2BGR);
	std::vector<std::vector<cv::Point> > hull(1);
	hull.push_back(stairsConvexHull);
	for (int i = 0; i < hull.size(); ++i)
	{
		cv::Scalar color = cv::Scalar(cv::theRNG().uniform(0, 255), cv::theRNG().uniform(0, 255), cv::theRNG().uniform(0, 255));
		cv::drawContours(temp, hull, i, color, 3);
	}

	imshow(std::to_string(cv::getTickCount()), temp);
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
	cv::Mat threshold;
	cv::blur(filter, threshold, cv::Size(3, 3));
	cv::threshold(threshold, threshold, thresh, maxval, type);

	int erosion_size = 1;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));

	/// Apply the erosion operation
	dilate(threshold, threshold, element);

	src.setTo(0, threshold);
}

void StairDetection::Probabilistic_Hough(cv::InputArray src, cv::OutputArray output) {
	HoughLinesP(src, output, 3, CV_PI / 720.0 * 1.0, min_Houghthreshold + houghThreshold, min_HoughLinelength, min_HoughLinegap);
}

/// groups the angles into one similar angle.
int StairDetection::GroupAngles(int angle)
{
	/// ignore angles between this range.
	/// because stairs shouldn't be perpendicular to the user.
	if (angle > 59 & angle < 122)
		return -1;

	/// shift the angle into the middle angle.
	int binSize = 10;
	int mod = angle % binSize;

	if (mod < binSize / 2.0)
		angle -= mod;
	else if (mod > binSize / 2.0)
		angle += mod;

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

		if (!vec.empty() && vec.size() > max) {
			stairsAngle = angle;
			max = vec.size();
		}
	}
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2)
{
	cv::Point2f r;
	cv::Point2f x = o2 - o1;
	cv::Point2f d1 = p1 - o1;
	cv::Point2f d2 = p2 - o2;

	float cross = d1.x*d2.y - d1.y*d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;

	if (r.x < 0 || r.x > 640)
		return false;
	if (r.y < 0 || r.y > 480)
		return false;

	return true;
}

/// From confident stair points,
/// find the best fit line that represents stairs.
void StairDetection::GetStairMidLine(std::vector<cv::Vec4i> &allLines, std::vector<int> &stairIndexes, cv::Vec4f &stairMidLine)
{
	std::vector<cv::Point2f> points;

	for (int index : stairIndexes) {
		cv::Vec4i &l = allLines[index];

		cv::Point p1(l[0], l[1]);
		cv::Point p2(l[2], l[3]);

		points.push_back(p1);
		points.push_back(p2);
	}

	if (!points.empty())
		cv::fitLine(points, stairMidLine, CV_DIST_L2, 0, 0.01, 0.01);
}

/// Using the found best fit line that represents the stairs,
/// find all lines that intersect with the fit line
/// all lines that intersect belong to the stairs
void StairDetection::GetStairPoints(std::vector<cv::Vec4i> &allLines, cv::Vec4f &stairMidLine, int &stairsAngle, std::vector<cv::Point> &stairPoints)
{
	cv::Point pt1, pt2;
	double theta = stairsAngle * CV_PI / 180;
	double a = cos(theta), b = sin(theta);

	pt1.x = cvRound(stairMidLine[2] + 640 * -b);
	pt1.y = cvRound(stairMidLine[3] + 480 * a);
	pt2.x = cvRound(stairMidLine[2] - 640 * -b);
	pt2.y = cvRound(stairMidLine[3] - 480 * a);

	for (cv::Vec4i vec : allLines) {
		cv::Point l1(vec[0], vec[1]);
		cv::Point l2(vec[2], vec[3]);

		if (intersection(l1, l2, pt1, pt2)) {
			stairPoints.push_back(l1);
			stairPoints.push_back(l2);
		}
	}
}

void StairDetection::ExtractStairsHull(std::vector<cv::Point> &stairPoints, std::vector<cv::Point> &stairsHull)
{
	if (!stairPoints.empty())
		cv::convexHull(cv::Mat(stairPoints), stairsHull);
}