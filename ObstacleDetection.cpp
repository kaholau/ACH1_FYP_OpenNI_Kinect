#include "ObstacleDetection.h"


ObstacleDetection::ObstacleDetection(int userHeight)
	: serial("COM7")
{
	mUserHeight = userHeight;
};

ObstacleDetection::~ObstacleDetection()
{
}

void ObstacleDetection::run(Mat* depth8bit, Mat* depth16bit, int angle)
{
	DepthMatSize = depth8bit->size();
	DepthMatRow = depth8bit->rows;
	DepthMatCol = depth8bit->cols;
	CameraAngle = (CameraAngle + angle) / 2;
	currentDepth8bit = depth8bit->clone();
	currentDepth16bit = depth16bit->clone();
	ObstacleList.clear();	
	GroundMaskCreate();	
	Segmentation(*depth8bit);

}



void ObstacleDetection::getOutputDepthImg(Mat *depth)
{	
	currentDepth8bit.copyTo(*depth);
}


void ObstacleDetection::Segmentation(Mat& depth8bit)
{
	Mat hist = HistogramCal(depth8bit);
	vector<int> LocalMinima = HistogramLocalMinima(hist);	
	SegementLabel(depth8bit, LocalMinima);
}

int ObstacleDetection::getColorIndex(int pixelValue, int index[], int indexSize){

	if (pixelValue < index[0])
		return 0;

	for (int i = 1; i < indexSize; i++)
	{
		if (index[i - 1] <= pixelValue&&pixelValue < index[i])
			return i;
	}
	return indexSize;
}

Mat ObstacleDetection::HistogramCal(Mat& img)
{
	int depth = 8;
	if (img.depth() == CV_32F)
		depth = 32;
	if (img.depth() == CV_16U)
		depth = 16;

	int numOfbin = HistSize;
	float range[] = { 0, (float)pow(2,depth) };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	Mat hist;
	calcHist(&img, 1, 0, Mat(), hist, 1, &numOfbin, &histRange, uniform, accumulate);

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
	imshow("histImgRaw", histImg);
#endif
	return hist;
}

vector<int> ObstacleDetection::HistogramLocalMinima(Mat& hist)
{

	struct buffer{
		struct buffer *prev;
		float data=0;
		struct buffer *next;
	};
	struct buffer node1,node2,node3;
	node1.next = &node2; node2.next = &node3; node3.next = &node1;
	node3.prev = &node2; node2.prev = &node1; node1.prev = &node3;
	struct buffer* current = &node1;
	vector<int> localMinIndex;
	localMinIndex.push_back(Valid_Distance);
	for (int i = Valid_Distance; i < HistSize - TooFarDistance; i++)
	{
		current->data = hist.at<float>(i) -hist.at<float>(i - 1);
		if ((current->next->data < 0) && (current->data >= 0) && (current->prev->data < 0))
			localMinIndex.push_back(i);
		current = current->next;

	}
	localMinIndex.push_back(HistSize - TooFarDistance);

#ifdef DISPLAY_HIST
	int histSize = hist.rows;
	int hist_w = 512; int hist_h = 400;
	int bin_w = 512 / histSize;
	Mat histImg(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	normalize(hist, hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImg, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(0, 0, 0), 2, 8, 0);
		if (i % 10 == 0)
			putText(histImg, std::to_string(i), Point(bin_w*i - 5, hist_h - 10), FONT_HERSHEY_PLAIN, 0.5, Scalar(128, 128, 128), 1);

		
	}

	for (int j = 0; j < localMinIndex.size(); j++)
	{
		
		line(histImg, Point(bin_w*(localMinIndex[j]), hist_h - cvRound(hist.at<float>(localMinIndex[j]) - 10)),
			Point(bin_w*(localMinIndex[j]), hist_h - cvRound(hist.at<float>(localMinIndex[j]) + 10)),
				Scalar(0, 0, 0), 2, 8, 0);

	}
#ifdef record_hist
	time_t timer;
	imwrite("C:/Users/HOHO/Pictures/sample/result/histogram/" + std::to_string(time(&timer)) + ".bmp", histImg);
#endif
	imshow("histImg", histImg);
#endif

	return localMinIndex;
}




void ObstacleDetection::SegementLabel(Mat& src, vector<int> &localMin)
{
	//for loop that contains vector is very slow, so convert the data into array
	int numOfSegement = localMin.size();
	int* localMinArray = new int[numOfSegement];
	for (int i = 0; i < numOfSegement; i++)
		localMinArray[i] = localMin[i];

	//seperate the segemeted region to different Mat according to the intervals among local minima
	//the 0th Image is alaways black with no segment and the last one will always be the ignore segment
	Mat* pThreasholdImageList = new Mat[numOfSegement-1];
	for (int i = 0; i < numOfSegement-1; i++)
		pThreasholdImageList[i] = Mat::zeros(src.size(), CV_8UC1); //Scalar(0) means whole img is black


	for (int r = 0; r < src.rows; r++)
	{
		for (int c = 0; c < src.cols; c++)
		{
			Scalar intensity = src.at<uchar>(r, c);
			if (intensity.val[0] != 0)
			{
				int index = getColorIndex((int)intensity.val[0], localMinArray, numOfSegement);
				//the 0th Image is alaways black with no segment and the last one will always the ignore segment
				if (index > 0 && index < numOfSegement)
					pThreasholdImageList[index - 1].at<uchar>(r, c) = 255;//255 is white
			}
			
		}
	}

	Mat obstacleMask (src.size(), CV_8UC1, Scalar(255));
	Mat MaskLayer1(src.size(), CV_8UC1, Scalar(255));
	Mat MaskLayer2(src.size(), CV_8UC1, Scalar(255));
	for (int i = 1; i <= 7; i+=2){
		rectangle(MaskLayer1, Point(src.cols*i / 7, 0), Point(src.cols * (i+1) / 7, src.rows), 0, CV_FILLED, 8, 0);
	}
	bitwise_not(MaskLayer1, MaskLayer2);

	//draw conuter to eliminate small segement, the 0th Image is alaways black with no segment and the last one will always the ignore segment
	Mat temp;
	for (int i = 1; i < numOfSegement - 1; i++)
	{
		temp = (MaskLayer1&pThreasholdImageList[i]);
		obstacleDetect(temp, obstacleMask);

		temp = (MaskLayer2&pThreasholdImageList[i]);
		obstacleDetect(temp, obstacleMask);
		
	}
	
#ifdef FOR_REPORT
	Mat obstacleMasktemp = obstacleMask.clone();
	flip(obstacleMasktemp, obstacleMasktemp, 1);
	imshow("obstacleMask", obstacleMasktemp);
#endif
	bitwise_not(GroundMat, GroundMat);
	if (!GroundMat.empty())
	{

		GroundMat &= obstacleMask;
		bitwise_not(GroundMat, GroundMat);
		int path = findPathByMassCenter();

		if (currentPathDir != path)
		{
			currentPathDir = path;
			serial.SendDirection(path);
		}
		if (path != NO_PATH){
			Scalar groundIndicator = Scalar(0, 128, 0);
			cvtColor(currentDepth8bit, currentDepth8bit, CV_GRAY2RGB);
			for (int r = 0; r < GroundMat.rows; r++)
				for (int c = 0; c < GroundMat.cols; c++)
				{	
					if (GroundMat.at<uchar>(r, c) == 0){
						currentDepth8bit.at<Vec3b>(r, c)[0] = groundIndicator[0];
						currentDepth8bit.at<Vec3b>(r, c)[1] = groundIndicator[1];
						currentDepth8bit.at<Vec3b>(r, c)[2] = groundIndicator[2];
					}
				}
	
		}
		else
		{
			cvtColor(currentDepth8bit, currentDepth8bit, CV_GRAY2RGB);
		}
			
	}
	delete[] pThreasholdImageList;
	
#ifdef DISPLAY_HULL
	vector<Vec4i> hierarchy_HULL;
	Mat hulldisplay = Mat(currentDepth8bit.size(), currentDepth8bit.type());
	for (int i = 0; i < ObstacleList.size(); i++)
		drawContours(hulldisplay, ObstacleList[i].contour, 0,
		Scalar(theRNG().uniform(1, 254), theRNG().uniform(1, 254), theRNG().uniform(1, 254)), -1, 8, hierarchy_HULL, 0, Point());
	imshow("hulldisplay", hulldisplay);
	//waitKey();
#endif

#ifdef DISPLAY_HEIGHT
	Mat heightdisplay = Mat(currentDepth8bit.size(), currentDepth8bit.type(),Scalar(255,255,255));
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
		int y = (int)GetHeight(p.y, currentDepth16bit.at<ushort>(p));
		putText(heightdisplay, std::to_string(y), p, FONT_HERSHEY_PLAIN, 0.7, Scalar(0, 0, 255), 1);
		circle(heightdisplay, p, 1, Scalar(0, 0, 255), 3);
	//	putText(currentColor, std::to_string(y), p, FONT_HERSHEY_PLAIN, 1.1, Scalar(0, 0, 255), 1);
	//	circle(currentColor, p, 1, Scalar(0, 0, 255), 3);
		flip(heightdisplay, heightdisplay, 1);
		imshow("heightdisplay", heightdisplay);
	}
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
		putText(currentDepth8bit, std::to_string(currentDepth8bit.at<uchar>(p)), p, FONT_HERSHEY_PLAIN, 0.7, Scalar(0, 0, 255), 1);
		circle(currentDepth8bit, p, 1, Scalar(0, 0, 255), 3);
	}
#endif	

#ifdef DISPLAY_DIR_LINE

	int	realDepthColsWidth = GroundMat.cols - 20 - 2;
	int realColStart = 20;
	int realColEnd = GroundMat.cols - 2;

	line(currentDepth8bit, Point(realColStart + realDepthColsWidth/4, currentDepth8bit.rows),
		Point(realColStart + realDepthColsWidth / 4, 0),
		Scalar(255), 2, 8, 0);
	line(currentDepth8bit, Point(realColStart + realDepthColsWidth *5/ 12, currentDepth8bit.rows),
		Point(realColStart + realDepthColsWidth * 5 / 12, 0),
		Scalar(255), 2, 8, 0);
	line(currentDepth8bit, Point(realColStart + realDepthColsWidth *7/ 12, currentDepth8bit.rows),
		Point(realColStart + realDepthColsWidth * 7 / 12, 0),
		Scalar(255), 2, 8, 0);
	line(currentDepth8bit, Point(realColStart + realDepthColsWidth*3 / 4, currentDepth8bit.rows),
		Point(realColStart + realDepthColsWidth * 3 / 4, 0),
		Scalar(255), 2, 8, 0);
	Scalar dir = Scalar(0, 0, 0);
	if (currentPathDir ==TURN_LEFT)
		dir = Scalar(0, 0, 255);
	if (currentPathDir == MIDDLE_LEFT)
		dir = Scalar(0, 255,255);
	if (currentPathDir == MIDDLE_MIDDLE)
		dir = Scalar(0, 255, 0);
	if (currentPathDir == MIDDLE_RIGHT)
		dir = Scalar(255, 255, 0);
	if (currentPathDir == TURN_RIGHT)
		dir = Scalar(255, 0, 0);
	arrowedLine(currentDepth8bit, Point(currentPathDirCol, currentDepth8bit.rows), Point(currentPathDirCol, currentDepth8bit.rows-20), dir, 3, 8, 0, 0.6);
#endif
	flip(currentDepth8bit, currentDepth8bit, 1);
}

void ObstacleDetection::obstacleDetect(Mat& img, Mat& output)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Find contours, need to read Ramer–Douglas–Peucker algorithm
	findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (vector<vector<Point>>::iterator it = contours.begin(); it != contours.end();)
	{
		if (it->size() < obstacle_size_ignore)
			it = contours.erase(it);
		else
			++it;
	}
		
	//cal the hull after reduce the contour's number to save time for hull
	vector<vector<Point> >hull(contours.size());
	/*vector<Moments> mu(contours.size());
	vector<Point2f> mc(contours.size());*/
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

float ObstacleDetection::GroundDirection(Vec3f& vector)
{
	//return atan(vector.val[0] / vector.val[1]);
	return (float)atan(sqrt(vector.val[0] * vector.val[0] + vector.val[2] * vector.val[2])/abs(vector.val[1])  );
}

void ObstacleDetection::GroundMaskCreate()
{
	int m_edge = planeEdgeForPlaneRemove;
	//(col,row)=(x,y)
	Point startPoint = Point(0, DepthMatRow / 2 - 1);
	if (GroundMat.empty())
	{
		GroundMat = Mat(DepthMatSize, CV_8UC1, Scalar(255));
		GroundBoolMat = Mat(DepthMatRow / m_edge, DepthMatCol / m_edge, CV_8UC1, Scalar(0));
	}
	else
	{
		GroundMat.setTo(Scalar(255));
		GroundBoolMat.setTo(Scalar(0));
	}

	float vec1_y = (float)(startPoint.y - startPoint.y - m_edge / 2);
	float vec2_y = (float)(startPoint.y + m_edge - startPoint.y - m_edge / 2);
	for (int center_r = startPoint.y + m_edge / 2, pt1_r = startPoint.y, pt2_r = startPoint.y + m_edge, interval_r=0;
		center_r < DepthMatRow&&pt1_r < DepthMatRow&&pt2_r < DepthMatRow;
		center_r += m_edge, pt1_r += m_edge, pt2_r += m_edge, interval_r += m_edge)
	{
		for (int center_c = startPoint.x + m_edge / 2, pt1_c = startPoint.x + m_edge, pt2_c = startPoint.x + m_edge,interval_c=0;
			center_c < DepthMatCol&&pt1_c < DepthMatCol&&pt2_c < DepthMatCol;
			center_c += m_edge, pt1_c += m_edge, pt2_c += m_edge, interval_c += m_edge)
		{
			Vec3f vec1 = { (float)(pt1_c - center_c), vec1_y, (float)(currentDepth16bit.at<ushort>(pt1_r, pt1_c) - currentDepth16bit.at<ushort>(center_r, center_c)) };
			Vec3f vec2 = { (float)(pt2_c - center_c), vec2_y, (float)(currentDepth16bit.at<ushort>(pt2_r, pt2_c) - currentDepth16bit.at<ushort>(center_r, center_c)) };
			Vec3f	crossProduct = vec2.cross(vec1);
			GroundBoolMatFill(Point(center_c, center_r), crossProduct);
			
#ifdef DISPLAY_ARROW
			Mat whitePaper = Mat(DepthMatRow * 4, DepthMatCol * 4, CV_8UC1, Scalar(255));
			putText(whitePaper, std::to_string(CameraAngle), Point(50, 50), FONT_HERSHEY_PLAIN, 0.9, Scalar(128), 1);
			GroundArrowDrawOnWhitePaper(whitePaper, crossProduct, Point(center_c + interval_c * 2, center_r + interval_r*2));
#endif

		}

	}
	GroundBoolMatToGroundMat();
	bitwise_and(currentDepth8bit, GroundMat, currentDepth8bit);
#ifdef FOR_REPORT
	Mat temp = GroundMat.clone();
	flip(temp, temp, 1);
	Mat temp1 = img.clone();
	flip(temp1, temp1, 1);
	imshow("ground mask", temp);
	imshow("img mask", temp1);
#endif

#ifdef DISPLAY_ARROW
	imshow("whitePaper", whitePaper);
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
	if ((vector.val[0]!=0 || vector.val[2]!=0) && vector.val[1]<0)
	{
		float angle = GroundDirection(vector);
		if (angle > minThreashold_horizontalPlane && angle < maxThreashold_horizontalPlane)
			if (h < Ground_height && h != -1 && h>GROUND_MIN_HIEGHT)
			{
				Point pt(location.x - planeEdgeForPlaneRemove / 2, location.y - planeEdgeForPlaneRemove / 2);
				GroundBoolMat.at<uchar>(pt.y / planeEdgeForPlaneRemove, pt.x / planeEdgeForPlaneRemove) = 255;
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
	for (int i = 0; i < contours.size(); i++)
		if (contours[i].size()>contours[maxIndex].size())
			maxIndex = i;
#ifdef FOR_REPORT
	Mat temp1 = Mat(boolMat.size(), boolMat.type(), Scalar(0));
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
				int edge = planeEdgeForPlaneRemove;
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

	if (holeDetectedInOneFrame)
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
	
	if (mu[maxAreaIndex].m00 < (planeAreaForPlaneRemove*(DILATION_SIZE * 2 + 1)*(DILATION_SIZE * 2 + 1)))
		return NO_PATH;
	int currentPathDirCol = mu[maxAreaIndex].m10 / mu[maxAreaIndex].m00;
	if ((currentPathDirCol > 0) && (currentPathDirCol <= realColStart + realDepthColsWidth / 4))
		return TURN_RIGHT;

	if ((currentPathDirCol > realColStart + realDepthColsWidth* 3/ 12) && (currentPathDirCol <= realColStart + realDepthColsWidth * 5 / 12))
		return MIDDLE_RIGHT;
	if ((currentPathDirCol > realColStart + realDepthColsWidth* 5/ 12) && (currentPathDirCol <= realColStart + realDepthColsWidth * 7 / 12))
		return MIDDLE_MIDDLE;
	if ((currentPathDirCol > realColStart + realDepthColsWidth* 7 / 12) && (currentPathDirCol <= realColStart + realDepthColsWidth * 9 / 12))
		return MIDDLE_LEFT;

	if ((currentPathDirCol > realColStart + realDepthColsWidth * 9 / 12) && (currentPathDirCol < realColStart + realDepthColsWidth))
		return TURN_LEFT;

	return NO_PATH;
}

