#include "ObstacleDetection.h"


ObstacleDetection::ObstacleDetection(int userHeight)
	: serial("COM7")
{
	mUserHeight = userHeight;
	DepthMatSize = Size(320, 240);
};


ObstacleDetection::~ObstacleDetection()
{
}

void ObstacleDetection::init(Size depthResolution)
{
	DepthMatSize = depthResolution;
	DepthMatRow = DepthMatSize.height;
	DepthMatCol = DepthMatSize.width;
	/*init for ground detection*/
	GroundMat = Mat(DepthMatSize, CV_8UC1, Scalar(255));
	GroundBoolMat = Mat(DepthMatRow / SQUARE_PLANE_EDGE, DepthMatCol / SQUARE_PLANE_EDGE, CV_8UC1, Scalar(0));
	/*initialise for obstacle detection*/
	pThreasholdImageList = new Mat[MAX_NUM_LOCALMINMA];
	for (int i = 0; i < MAX_NUM_LOCALMINMA; i++)
		pThreasholdImageList[i] = Mat::zeros(DepthMatSize, CV_8UC1); //Scalar(0) means whole img is black
	obstacleMask = Mat(DepthMatSize, CV_8UC1, Scalar(255));
	MaskLayer1 = Mat(DepthMatSize, CV_8UC1, Scalar(255));
	MaskLayer2 = Mat(DepthMatSize, CV_8UC1, Scalar(255));
	for (int i = 1; i <= 5; i += 2)
		rectangle(MaskLayer1, Point(DepthMatCol*i / 5, 0), Point(DepthMatCol * (i + 1) / 5, DepthMatRow), 0, CV_FILLED, 8, 0);

	bitwise_not(MaskLayer1, MaskLayer2);
}
void ObstacleDetection::run(Mat* depth8bit, Mat* depth16bit, int angle)
{
	
	CameraAngle = (CameraAngle + angle) / 2;
	outputDepth8bit = depth8bit->clone();
	currentDepth8bit = depth8bit->clone();
	currentDepth16bit = depth16bit->clone();

#ifdef DISPLAY_HULL
	ObstacleList.clear();	
#endif
	GroundMaskCreate();	
	Segmentation();
	output();

}


void ObstacleDetection::GroundMaskCreate()
{
	int m_edge = SQUARE_PLANE_EDGE;
	//(col,row)=(x,y)
	Point startPoint = Point(0, DepthMatRow / 2 - 1);
	GroundMat.setTo(Scalar(255));
	GroundBoolMat.setTo(Scalar(0));

#ifdef DISPLAY_ARROW
	Mat whitePaper = Mat(DepthMatSize * 3, CV_8UC1, Scalar(255));
#endif
#ifdef DISPLAY_HEIGHT
	Mat heightdisplay = Mat(DepthMatSize * 4, CV_8UC1, Scalar(255));
#endif
	float vec1_y = (float)(startPoint.y - startPoint.y - m_edge / 2);
	float vec2_y = (float)(startPoint.y + m_edge - startPoint.y - m_edge / 2);
	for (int center_r = startPoint.y + m_edge / 2, pt1_r = startPoint.y, pt2_r = startPoint.y + m_edge, interval_r = 0;
		center_r < DepthMatRow&&pt1_r < DepthMatRow&&pt2_r < DepthMatRow;
		center_r += m_edge, pt1_r += m_edge, pt2_r += m_edge, interval_r += m_edge)
	{
		for (int center_c = startPoint.x + m_edge / 2, pt1_c = startPoint.x + m_edge, pt2_c = startPoint.x + m_edge, interval_c = 0;
			center_c < DepthMatCol&&pt1_c < DepthMatCol&&pt2_c < DepthMatCol;
			center_c += m_edge, pt1_c += m_edge, pt2_c += m_edge, interval_c += m_edge)
		{
			Vec3f vec1 = { (float)(pt1_c - center_c), vec1_y, (float)(currentDepth16bit.at<ushort>(pt1_r, pt1_c) - currentDepth16bit.at<ushort>(center_r, center_c)) };
			Vec3f vec2 = { (float)(pt2_c - center_c), vec2_y, (float)(currentDepth16bit.at<ushort>(pt2_r, pt2_c) - currentDepth16bit.at<ushort>(center_r, center_c)) };
			Vec3f	crossProduct = vec2.cross(vec1);
			GroundBoolMatFill(Point(center_c, center_r), crossProduct);

#ifdef DISPLAY_ARROW
			putText(whitePaper, std::to_string(CameraAngle), Point(50, 50), FONT_HERSHEY_PLAIN, 0.9, Scalar(128), 1);
			GroundArrowDrawOnWhitePaper(whitePaper, crossProduct, Point(center_c + interval_c * 2, center_r + interval_r * 2));
#endif
#ifdef DISPLAY_HEIGHT
			int h = (int)GetHeight(center_r, currentDepth16bit.at<ushort>(Point(center_c , center_r  )));
			putText(heightdisplay, std::to_string(h), Point(center_c + interval_c * 2.8, center_r + interval_r * 2.8), FONT_HERSHEY_PLAIN, 0.7, Scalar(0, 0, 255), 1);
			circle(heightdisplay, Point(center_c + interval_c * 2.8, center_r + interval_r * 2.8), 0.5, Scalar(50), 3);
#endif

		}

	}
	GroundBoolMatToGroundMat();
	bitwise_and(currentDepth8bit, GroundMat, currentDepth8bit);
#ifdef FOR_REPORT
	Mat temp = GroundMat.clone();
	flip(temp, temp, 1);
	Mat temp1 = currentDepth8bit.clone();
	flip(temp1, temp1, 1);
	imshow("ground mask", temp);
	imshow("img mask", temp1);
#endif

#ifdef DISPLAY_ARROW
	imshow("whitePaper", whitePaper);
#endif
#ifdef DISPLAY_HEIGHT
	imshow("heightdisplay", heightdisplay);
#endif

}



/*parameter:
img : image contains result of cross product
*/
void ObstacleDetection::GroundBoolMatFill(Point& location, Vec3f& vector)
{
	/*
	Testing the height of the surface
	*/
	int h = (int)GetHeight(location.y, currentDepth16bit.at<ushort>(location));
	if (h < GROUND_MIN_HIEGHT&&!holeDetectedInOneFrame)
	{
		holeDetectedCountInOneFrame++;
		if (holeDetectedCountInOneFrame > HOLE_DETECED_ONE_CONFIRM_COUNT)
		{
			holeDetectedInOneFrame = true;
			//std::cout << " holeDetectedInOneFrame ";
		}
	}
	/*
	Testing the surface normal angle of the horizontal plane,from experiment, ground's vector has -ve y coordinate
	*/
	if ((vector.val[0] != 0 || vector.val[2] != 0) && vector.val[1]<0)
	{
		float angle = GroundDirection(vector);
		if (angle > MIN_TH_NORMAL && angle < MAX_TH_NORMAL)
			if (h < GROUND_HEIGHT && h != -1 && h>GROUND_MIN_HIEGHT)
			{
				Point pt(location.x - SQUARE_PLANE_EDGE / 2, location.y - SQUARE_PLANE_EDGE / 2);
				GroundBoolMat.at<uchar>(pt.y / SQUARE_PLANE_EDGE, pt.x / SQUARE_PLANE_EDGE) = 255;
			}
	}
}
void ObstacleDetection::GroundBoolMatToGroundMat()
{
	/*
	Calculate the broundary of the Boolean Mat
	*/
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	int maxIndex = 0;
	Mat temp = GroundBoolMat.clone();
	findContours(temp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	//imshow("bool", boolMat);
	if (contours.size() < 1)	return;
	//vector<vector<Point> >hull(contours.size());
	
	for (int i = 0; i < contours.size(); i++)
		if (contours[i].size()>contours[maxIndex].size())
			maxIndex = i;
	//convexHull(Mat(contours[maxIndex]), hull[maxIndex], false);
#ifdef FOR_REPORT
	Mat temp1 = Mat(GroundBoolMat.size(), GroundBoolMat.type(), Scalar(0));
	drawContours(temp1, contours, maxIndex, Scalar(255), 1, 8);
	imshow("OLDtemp", temp1);
#endif
	/*
	Apply dilation on the Boolean Mat
	*/
	Mat elementD = getStructuringElement(MORPH_RECT, Size(2 * DILATION_SIZE + 1, 2 * DILATION_SIZE + 1), Point(DILATION_SIZE, DILATION_SIZE));
	dilate(GroundBoolMat, GroundBoolMat, elementD);

	/*
	Convert to GroundMat
	*/
	for (int i = 0; i < GroundBoolMat.rows; i++)
		for (int j = 0; j < GroundBoolMat.cols; j++)
			if ((GroundBoolMat.at<uchar>(i, j) == 255) && pointPolygonTest(contours[maxIndex], Point(j, i), 0) != -1)
				//if ((boolMat.at<uchar>(i, j) == 255))
			{
				int edge = SQUARE_PLANE_EDGE;
				Point center((j*edge) + (edge / 2), (i * edge) + (edge - 1) + (edge / 2));
				Point pt1(center.x - edge / 2, center.y - edge / 2), pt2(center.x + edge / 2, center.y + edge / 2);
				rectangle(GroundMat, pt1, pt2, Scalar(0), CV_FILLED);
			}
}

int ObstacleDetection::findHole()
{
	if (!holeDetectedInOneFrame)
	{
		holeDetectedCountInOneFrame = 0;
		if (holeDetectedCount > 0)
		{
			holeDetectedCount = 0; holeDetected = false;
			//std::cout << " reset_holeDetected " << std::endl;
		}

	}
	else
	{
		//reset holeDetectedInOneFrame,holeDetectedCountInOneFrame
		holeDetectedInOneFrame = false;
		holeDetectedCountInOneFrame = 0;
		holeDetectedCount++;

		if (holeDetectedCount > HOLE_DETECED_CONFIRM_COUNT)
		{
			holeDetected = true; holeDetectedCount = 0;
			//std::cout << " holeDetected "<<std::endl;
		}
	}

	if (holeDetected&&!angleSetToLookDown)
	{
		string speech = HOLE_DETECTED_SPEECH;
		TextToSpeech::pushBack(speech);
		angleSetToLookDown = true;
		std::cout << " lookdown" << std::endl;
		return MOTOR_LOOK_DOWN;
	}

	if (!holeDetected&&angleSetToLookDown)
	{
		string speech = NO_HOLE_DETECTED_SPEECH;
		TextToSpeech::pushBack(speech);
		angleSetToLookDown = false;
		std::cout << " lookup " << std::endl;
		return INIT_CAMERA_ANGLE;
	}

	return 99;//out of valid range
}
/*parameter:
img : image to be draw
vector : the vector of normal
start: the arrow's starting point
*/
void ObstacleDetection::GroundArrowDraw(Mat& img, Vec3f& vector, Point& start)
{

	//cal unit vector<x,y,z>=<col,row,depth>
	float lenght = sqrt(vector.val[0] * vector.val[0] + vector.val[1] * vector.val[1] + vector.val[2] * vector.val[2]);
	vector.val[0] = vector.val[0] / lenght;
	vector.val[1] = vector.val[1] / lenght;
	vector.val[2] = vector.val[2] / lenght;
	vector *= 15;
	//cal point of arrow	//(col,row)=point(x,y)
	Point end = Point((int)(start.x + vector.val[0]), (int)(start.y + vector.val[1]));
	arrowedLine(img, start, end, Scalar(255), 1, 8, 0, 0.5);
}

/*
Draw on a  Mat with white background
parameter:
img : image to be draw
vector : the vector of normal
start: the arrow's starting point
*/
void ObstacleDetection::GroundArrowDrawOnWhitePaper(Mat& img, Vec3f& vector, Point& start)
{
	//cal unit vector<x,y,z>=<col,row,depth>
	float lenght = sqrt(vector.val[0] * vector.val[0] + vector.val[1] * vector.val[1] + vector.val[2] * vector.val[2]);
	vector.val[0] = vector.val[0] / lenght;
	vector.val[1] = vector.val[1] / lenght;
	vector.val[2] = vector.val[2] / lenght;
	vector *= 15;
	//cal point of arrow	//(col,row)=point(x,y)
	Point end;
	if (vector.val[0]>0)
		end = Point((int)(start.x + sqrt(vector.val[0] * vector.val[0] + vector.val[2] * vector.val[2])), (int)(start.y + vector.val[1]));
	else
		end = Point((int)(start.x - sqrt(vector.val[0] * vector.val[0] + vector.val[2] * vector.val[2])), (int)(start.y + vector.val[1]));
	arrowedLine(img, start, end, Scalar(0), 2, 8, 0, 0.4);


	std::stringstream stream;
	stream << std::fixed << std::setprecision(2) << GroundDirection(vector);
	string s = stream.str();
	putText(img, s, start, FONT_HERSHEY_PLAIN, 0.8, Scalar(150), 1);
}


float ObstacleDetection::GroundDirection(Vec3f& vector)
{
	//return atan(vector.val[0] / vector.val[1]);
	return (float)atan(sqrt(vector.val[0] * vector.val[0] + vector.val[2] * vector.val[2]) / abs(vector.val[1]));
}


void ObstacleDetection::Segmentation()
{
	vector<int> LocalMinima = HistogramCal();
	SegementLabel(LocalMinima);
}

int ObstacleDetection::getColorIndex(int pixelValue, int index[], int indexSize){

	if (pixelValue < index[0])	return 0;
	for (int i = 1; i < indexSize; i++)
		if (index[i - 1] <= pixelValue&&pixelValue < index[i])	return i;
	return indexSize;
}

vector<int> ObstacleDetection::HistogramCal()
{
	int depth = 8;
	int numOfbin = HISTOGRAM_SIZE;
	float range[] = { 0, (float)pow(2,depth) };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	Mat hist;
	calcHist(&currentDepth8bit, 1, 0, Mat(), hist, 1, &numOfbin, &histRange, uniform, accumulate);

#ifdef DISPLAY_HIST
	int histSize = hist.rows;
	int hist_w = 512; int hist_h = 400;
	int bin_w = 512 / histSize;
	Mat histImg(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	Mat tempHist = hist.clone();
	normalize(tempHist, tempHist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImg, Point(bin_w*(i - 1), hist_h - cvRound(tempHist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(tempHist.at<float>(i))),
			Scalar(0, 0, 0), 2, 8, 0);
		if (i % 10 == 0)
			putText(histImg, std::to_string(i), Point(bin_w*i - 5, hist_h - 10), FONT_HERSHEY_PLAIN, 0.5, Scalar(128, 128, 128), 1);
	}
	
#endif

	GaussianBlur(hist, hist, Size(1, 17), 0, 0);
#ifdef DISPLAY_HIST
	tempHist = hist.clone();
	normalize(tempHist, tempHist, 0, histImg.rows, NORM_MINMAX, -1, Mat());
	for (int i = 1; i < histSize; i++)
	{
		line(histImg, Point(bin_w*(i - 1), hist_h - cvRound(tempHist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(tempHist.at<float>(i))),
			Scalar(100, 0, 0), 2, 8, 0);
		
	}
	//imshow("histImgRaw", histImg);
#endif
	
	struct buffer{
		struct buffer *prev;
		float data = 0;
		struct buffer *next;
	};
	struct buffer node1, node2, node3;
	node1.next = &node2; node2.next = &node3; node3.next = &node1;
	node3.prev = &node2; node2.prev = &node1; node1.prev = &node3;
	struct buffer* current = &node1;
	vector<int> localMinIndex;
	/* 0 means invalid in depth*/
	localMinIndex.push_back(1);
	for (int i = 1; i < HISTOGRAM_SIZE - TOO_FAR_PIXEL_VALUE; i++)
	{
		current->data = hist.at<float>(i) -hist.at<float>(i - 1);
		if ((current->next->data < 0) && (current->data >= 0) && (current->prev->data < 0))
			localMinIndex.push_back(i);
		current = current->next;
	}
	localMinIndex.push_back(HISTOGRAM_SIZE - TOO_FAR_PIXEL_VALUE);

#ifdef DISPLAY_HIST

	for (int j = 0; j < localMinIndex.size(); j++)
	{

		line(histImg, Point(bin_w*(localMinIndex[j]), hist_h - cvRound(tempHist.at<float>(localMinIndex[j]) - 10)),
			Point(bin_w*(localMinIndex[j]), hist_h - cvRound(tempHist.at<float>(localMinIndex[j]) + 10)),
			Scalar(0, 0, 0), 2, 8, 0);
	}

	imshow("histImg", histImg);
#endif

	return localMinIndex;
}

void ObstacleDetection::output()
{

#ifdef FOR_REPORT
	Mat obstacleMasktemp = obstacleMask.clone();
	flip(obstacleMasktemp, obstacleMasktemp, 1);
	imshow("obstacleMask", obstacleMasktemp);
#endif

	int path = findPathByMassCenter();

	if (currentPathDir != path)
	{
		currentPathDir = path;
		serial.SendDirection(path);
	}
	

#ifdef DISPLAY_HULL
	vector<Vec4i> hierarchy_HULL;
	Mat hulldisplay = Mat(currentDepth8bit.size(), currentDepth8bit.type());
	for (int i = 0; i < ObstacleList.size(); i++)
		drawContours(hulldisplay, ObstacleList[i].contour, 0,
		Scalar(theRNG().uniform(1, 254), theRNG().uniform(1, 254), theRNG().uniform(1, 254)), -1, 8, hierarchy_HULL, 0, Point());
	imshow("hulldisplay", hulldisplay);
	//waitKey();
#endif

#ifdef DISPLAY_DISTANCE
	vector<Point> test;
	for (int y = 10; y < currentDepth8bit.rows; y += 24)
	{
		for (int x = 10; x < currentDepth8bit.cols; x += 32)
		{
			test.push_back(Point(x, y));
		}
	}

	for (Point p : test)
	{
		//putText(currentDepth8bit, std::to_string(currentDepth16bit.at<ushort>(p)), p, FONT_HERSHEY_PLAIN, 0.7, Scalar(0, 0, 255), 1);
		putText(outputDepth8bit, std::to_string(currentDepth16bit.at<uchar>(p)), p, FONT_HERSHEY_PLAIN, 0.7, Scalar(0, 0, 255), 1);
		circle(outputDepth8bit, p, 1, Scalar(0, 0, 255), 3);
	}
#endif	
#ifdef DISPLAY_DIR_LINE
	bitwise_not(GroundMat, GroundMat);
	GroundMat &= obstacleMask;
	bitwise_not(GroundMat, GroundMat);
	if (path != NO_PATH){
		Scalar groundIndicator = Scalar(0, 128, 0);
		cvtColor(outputDepth8bit, outputDepth8bit, CV_GRAY2RGB);
		for (int r = 0; r < GroundMat.rows; r++)
			for (int c = 0; c < GroundMat.cols; c++)
			{
				if (GroundMat.at<uchar>(r, c) == 0){
					outputDepth8bit.at<Vec3b>(r, c)[0] = groundIndicator[0];
					outputDepth8bit.at<Vec3b>(r, c)[1] = groundIndicator[1];
					outputDepth8bit.at<Vec3b>(r, c)[2] = groundIndicator[2];
				}
			}

	}
	else
	{
		cvtColor(outputDepth8bit, outputDepth8bit, CV_GRAY2RGB);
	}

	int	realDepthColsWidth = GroundMat.cols - 20 - 2;
	int realColStart = 20;
	int realColEnd = GroundMat.cols - 2;

	line(outputDepth8bit, Point(realColStart + right_line, outputDepth8bit.rows),
		Point(realColStart + right_line, 0),
		Scalar(255), 2, 8, 0);
	line(outputDepth8bit, Point(realColStart + right_middle_line, outputDepth8bit.rows),
		Point(realColStart + right_middle_line, 0),
		Scalar(255), 2, 8, 0);
	line(outputDepth8bit, Point(realColStart + left_middle_line, outputDepth8bit.rows),
		Point(realColStart + left_middle_line, 0),
		Scalar(255), 2, 8, 0);
	line(outputDepth8bit, Point(realColStart + left_line, outputDepth8bit.rows),
		Point(realColStart + left_line, 0),
		Scalar(255), 2, 8, 0);
	Scalar dir = Scalar(0, 0, 0);
	if (currentPathDir == TURN_LEFT)
		dir = Scalar(0, 0, 255);
	if (currentPathDir == MIDDLE_LEFT)
		dir = Scalar(0, 255, 255);
	if (currentPathDir == MIDDLE_MIDDLE)
		dir = Scalar(0, 255, 0);
	if (currentPathDir == MIDDLE_RIGHT)
		dir = Scalar(255, 255, 0);
	if (currentPathDir == TURN_RIGHT)
		dir = Scalar(255, 0, 0);
	arrowedLine(outputDepth8bit, Point(currentPathDirCol, outputDepth8bit.rows), Point(currentPathDirCol, outputDepth8bit.rows - 20), dir, 3, 8, 0, 0.6);
	flip(outputDepth8bit, outputDepth8bit, 1);
#endif

}

void ObstacleDetection::SegementLabel(vector<int> &localMin)
{
	//for loop that contains vector is very slow, so convert the data into array
	int numOfSegement = (localMin.size() > MAX_NUM_LOCALMINMA) ? MAX_NUM_LOCALMINMA: localMin.size();
	int* localMinArray = new int[numOfSegement];
	for (int i = 0; i < numOfSegement; i++)
		localMinArray[i] = localMin[i];
	/*
	seperate the segemeted region to different Mat according to the intervals among local minima
	the 0th Image is alaways black with no segment and the last one will always be the ignore segment
	*/
	for (int i = 0; i < numOfSegement - 1; i++)
		pThreasholdImageList[i].setTo(Scalar(0,0,0)); //Scalar(0) means whole img is black

	for (int r = 0; r < DepthMatRow; r++)
		for (int c = 0; c < DepthMatCol; c++)
		{
			Scalar intensity = currentDepth8bit.at<uchar>(r, c);
			if (intensity.val[0] != 0)
			{
				int index = getColorIndex((int)intensity.val[0], localMinArray, numOfSegement);
				//the 0th Image is alaways black with no segment and the last one will always the ignore segment
				if (index > 0 && index < numOfSegement)
					pThreasholdImageList[index - 1].at<uchar>(r, c) = 255;//255 is white
			}
		}
	delete[] localMinArray;

	obstacleMask.setTo(Scalar(255));
	
	//draw conuter to eliminate small segement, the 0th Image is alaways black with no segment and the last one will always the ignore segment
	Mat temp;
	for (int i = 1; i < numOfSegement - 1; i++)
	{
		temp = (MaskLayer1&pThreasholdImageList[i]);
		obstacleDetect(temp, obstacleMask);

		temp = (MaskLayer2&pThreasholdImageList[i]);
		obstacleDetect(temp, obstacleMask);
		
	}

}

void ObstacleDetection::obstacleDetect(Mat& img, Mat& output)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Find contours, need to read Ramer–Douglas–Peucker algorithm
	findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (vector<vector<Point>>::iterator it = contours.begin(); it != contours.end();)
	{
		if (it->size() < OBSTACLE_SIZE_IGNORE)
			it = contours.erase(it);
		else
			++it;
	}
		
	//cal the hull after reduce the contour's number to save time for hull
	vector<vector<Point> >hull(contours.size());
#ifdef DISPLAY_HULL
	vector<Moments> mu(contours.size());
	vector<Point2f> mc(contours.size());
#endif
	for (size_t i = 0; i < contours.size(); i++)
	{
		convexHull(Mat(contours[i]), hull[i], false);
		//Scalar color = Scalar(theRNG().uniform(1, 254), theRNG().uniform(1, 254), theRNG().uniform(1, 254));
		drawContours(output, hull, i, Scalar(0), -1, 8, hierarchy, 0, Point());
#ifdef DISPLAY_HULL
		mu[i] = moments(contours[i], false);

		//m00 is the zero moment which is the area of the contour
		if (mu[i].m00 == 0) continue;
		if (contourArea(contours[i]) / arcLength(contours[i], true) < 3) continue;


		mc[i] = Point2f((float)(mu[i].m10 / mu[i].m00), (float)(mu[i].m01 / mu[i].m00));
		createObstacle(hull[i], Point((int)mc[i].x, (int)mc[i].y));	
#endif
	}
		
}


void ObstacleDetection::createObstacle(vector<Point> contour, Point center)
{
	Object obj;
	obj.contour.push_back(contour);
	obj.pos = center;

	ObstacleList.push_back(obj);

}

double ObstacleDetection::GetPointAngle(const int pointY)
{
	//myfile << "CameraAngle: " << CameraAngle << std::endl;
	return CameraAngle + (double)(currentDepth8bit.rows / 2.0-(double)pointY) / (double)currentDepth8bit.rows * 43;
}

double ObstacleDetection::GetHeight(const int pointY, const int depth)
{
	/// PARAMETERS:
	/// 0 <= pointY < 480
	/// Depth in 16bit Raw format.

	// error
	if (depth == 0)
		return -1;

	double angleDeg = GetPointAngle(pointY);
	double temp = mUserHeight + depth * sin(angleDeg / 180 * CV_PI);
	//myfile << "mUserHeight: " << mUserHeight << " depth: " << depth << " pointY: " << pointY << "angleDeg=" << angleDeg << " Height=" << temp << std::endl;

	return temp;
}

int  ObstacleDetection::findPathByMassCenter()
{
	int realDepthColsWidth;
	int realColStart;
	int realColEnd;

	if (GroundMat.cols == 320)
	{
		realDepthColsWidth = GroundMat.cols - 20 - 2;
		realColStart = 20;
		realColEnd = GroundMat.cols - 2;
	}

	if (GroundMat.cols == 640)
	{
		realDepthColsWidth = GroundMat.cols - 45 - 5;
		realColStart = 45;
		realColEnd = GroundMat.cols - 5;
	}
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	Mat temp = GroundMat.clone();
	bitwise_not(temp, temp);
	//imshow("temp", temp);
	findContours(temp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	if (contours.size() <= 0)
		return NO_PATH;
	vector<Moments> mu(contours.size());
	int maxAreaIndex=0;
	
	//mu[i].m00 is area size when using binary image
	for (size_t i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
		if (mu[i].m00 > mu[maxAreaIndex].m00)
			maxAreaIndex = i;		
	}
	
	if (mu[maxAreaIndex].m00 < (SQUARE_PLANE_AREA*(DILATION_SIZE * 2 + 1)*(DILATION_SIZE * 2 + 1)))
		return NO_PATH;
	currentPathDirCol = mu[maxAreaIndex].m10 / mu[maxAreaIndex].m00;
	if ((currentPathDirCol > 0) && (currentPathDirCol <= realColStart + right_line))
		return TURN_RIGHT;

	if ((currentPathDirCol > realColStart + right_line) && (currentPathDirCol <= realColStart + right_middle_line))
		return MIDDLE_RIGHT;
	if ((currentPathDirCol > realColStart + right_middle_line) && (currentPathDirCol <= realColStart + left_middle_line))
		return MIDDLE_MIDDLE;
	if ((currentPathDirCol > realColStart + left_middle_line) && (currentPathDirCol <= left_line))
		return MIDDLE_LEFT;

	if ((currentPathDirCol > realColStart + left_line) && (currentPathDirCol < realColStart + realDepthColsWidth))
		return TURN_LEFT;

	return NO_PATH;
}


void ObstacleDetection::getOutputDepthImg(Mat *depth)
{
	outputDepth8bit.copyTo(*depth);
}

