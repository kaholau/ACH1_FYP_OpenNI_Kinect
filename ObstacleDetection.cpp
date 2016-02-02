#include "ObstacleDetection.h"


void ObstacleDetection::test()
{
#ifdef TEST_SEGMENTATION
	//----------------KaHO-----------
	char* filename = new char[150];
	sprintf(filename, SAMPLE_PATH);
	Mat src = imread(filename, CV_LOAD_IMAGE_COLOR);
	Mat gray;
	cvtColor(src, gray, CV_BGR2GRAY);
	//if ((gray.depth() == CV_8U) && (gray.channels() == 1))
	//	std::cout << "gray is 8UC1" << std::endl;
	

	run(&gray);
	getOutputDepthImg(&src);
	imshow("src", src);
	waitKey();
	//double t = (double)getTickCount();
	//t = ((double)getTickCount() - t) / getTickFrequency();
	//std::cout << " Total used : " << t << " seconds" << std::endl;

	//---------------KaHo----------------
#endif
}


ObstacleDetection::ObstacleDetection(int userHeight)
{
	mUserHeight = userHeight;
	myfile.open(MYFILE_PATH);
};

ObstacleDetection::~ObstacleDetection()
{
	myfile.close();
}

void ObstacleDetection::run(Mat* pImg)
{
	//double t = (double)getTickCount();
	GaussianBlur(*pImg, *pImg, Size(1, 13), 0, 0);
	currentDepth =pImg->clone();
	//ObstacleList.clear();
	
	
	//std::cout << " Total used : " << t << " seconds" << std::endl;
	GroundMaskCreate(*pImg);

	
	Segmentation(*pImg);
	/*t = ((double)getTickCount() - t) / getTickFrequency();
	OutputDebugString(L"time: ");
	OutputDebugStringA(std::to_string(t).c_str());
	OutputDebugString(L"\n");*/
}


void ObstacleDetection::setCurrentColor(Mat* pImg)
{
	currentColor = pImg->clone();
}

void ObstacleDetection::getOutputDepthImg(Mat *depth)
{	
	currentDepth.copyTo(*depth);
}

void ObstacleDetection::getOutputColorImg(Mat *color)
{	
	currentColor.copyTo(*color);
}

void ObstacleDetection::Segmentation(Mat& src)
{

	Mat hist = HistogramCal(src);	
	vector<int> LocalMinima = HistogramLocalMinima(hist);	
	SegementLabel(src, LocalMinima);
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
	GaussianBlur(hist, hist, Size(1, 17), 0, 0);

	/*int histSize = hist.rows;
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

	imshow("histImg", histImg);
	waitKey();*/

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
	for (int i = Valid_Distance; i < HistSize - 1; i++)
	{
		current->data = hist.at<float>(i) -hist.at<float>(i - 1);
		if ((current->next->data < 0) && (current->data >= 0) && (current->prev->data < 0))
			localMinIndex.push_back(i);
		current = current->next;

	}

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
	int numOfThreashold = localMin.size()-1;
	Mat* pThreasholdImageList = new Mat[localMin.size()+1];
	for (int i = 0; i < numOfThreashold; i++)
		pThreasholdImageList[i] = Mat::zeros(src.size(), CV_8UC1); //Scalar(0) means whole img is black


	for (int r = 0; r < src.rows; r++)
	{
		for (int c = 0; c < src.cols; c++)
		{
			Scalar intensity = src.at<uchar>(r, c);
			int index = getColorIndex((int)intensity.val[0], localMinArray, numOfSegement);
			//the 0th Image is alaways black with no segment and the last one will always the ignore segment
			if (index > 0 && index <= numOfThreashold)
				pThreasholdImageList[index-1].at<uchar>(r, c) = 255;//255 is white
		}
	}

	Mat obstacleMask (src.size(), CV_8UC1, Scalar(255));
	//draw conuter to eliminate small segement, the 0th Image is alaways black with no segment and the last one will always the ignore segment
	for (int i = 1; i < numOfThreashold - 1; i++)
	{
		obstacleDetect(pThreasholdImageList[i], obstacleMask);
		/*imshow("threashold", pThreasholdImageList[i]);
		waitKey();*/
	}
	//imshow("obstacleMask", obstacleMask);
	//waitKey();
	//bitwise_not(obstacleMask, obstacleMask);
	bitwise_not(Ground.img, Ground.img);
	if (!Ground.img.empty())
	{
		Ground.img &= obstacleMask;
		//createPlaneObject(currentDepth, Ground.img, GROUND);

		bitwise_not(Ground.img, Ground.img);
		std::string path = findPath();

		if (currentPath.compare(path) != 0)
		{
			currentPath = path;
			TextToSpeech::pushBack(path);
		}
		if (path.compare("no path") != 0)
			currentDepth &= Ground.img;
	}
	delete[] pThreasholdImageList;
	
	//cvtColor(currentDepth, currentDepth, CV_GRAY2RGBA);	

	//find path just using ground img
	//imshow("Ground.img",Ground.img);
	//waitKey();



#ifdef DISPLAY_HEIGHT

	for (size_t i = 0; i < ObstacleList.size(); i++)
		int y = (int)GetHeight(ObstacleList[i].pos.y, currentRawDepth.at<ushort>(ObstacleList[i].pos));

	vector<Point> test;
	for (int y = 10; y < currentDepth.rows; y += 50)
	{
		for (int x = 10; x < currentDepth.cols; x += 50)
		{
			test.push_back(Point(x, y));
		}
	}

	for (Point p : test)
	{
		int y = (int)GetHeight(p.y, currentRawDepth.at<ushort>(p) >> 3);
		putText(currentDepth, std::to_string(y), p, FONT_HERSHEY_PLAIN, 1.1, Scalar(0, 0, 255), 1);
		circle(currentDepth, p, 1, Scalar(0, 0, 255), 3);
	//	putText(currentColor, std::to_string(y), p, FONT_HERSHEY_PLAIN, 1.1, Scalar(0, 0, 255), 1);
	//	circle(currentColor, p, 1, Scalar(0, 0, 255), 3);

	}
#endif

#ifdef DISPLAY_HULL
	vector<Vec4i> hierarchy;
	for (int i = 0; i < ObstacleList.size();i++)
		drawContours(currentDepth, ObstacleList[i].contour, 0, 
			Scalar(theRNG().uniform(1, 254), theRNG().uniform(1, 254), theRNG().uniform(1, 254)), -1, 8, hierarchy, 0, Point());
	//imshow("currentDepth", currentDepth);
	//waitKey();
#endif
	
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
	vector<Moments> mu(contours.size());
	vector<Point2f> mc(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		convexHull(Mat(contours[i]), hull[i], false);
		//Scalar color = Scalar(theRNG().uniform(1, 254), theRNG().uniform(1, 254), theRNG().uniform(1, 254));
		drawContours(output, hull, i, Scalar(0), -1, 8, hierarchy, 0, Point());
		mu[i] = moments(contours[i], false);

		//m00 is the zero moment which is the area of the contour
		if (mu[i].m00 == 0) continue;
		if (contourArea(contours[i]) / arcLength(contours[i], true) < 3) continue;


		//mc[i] = Point2f((float)(mu[i].m10 / mu[i].m00), (float)(mu[i].m01 / mu[i].m00));
		//createObstacle(hull[i], OBSTACLE, Point((int)mc[i].x, (int)mc[i].y));	

	}
		
}



void ObstacleDetection::GroundMaskCreate(Mat &img)
{
	
	int edge = planeEdgeForPlaneRemove;

	//(col,row)=(x,y)
	//Point startPoint = Point(0, img.rows / 2 - 1);
	//myfile << "img.rows: " << img.rows << std::endl;
	Point startPoint = Point(0, edge-1);
	Ground.img = Mat(img.size(), CV_8UC1,Scalar(255));
	for (int center_r = startPoint.y + edge / 2, pt1_r = startPoint.y, pt2_r = startPoint.y + edge;
		center_r < img.rows&&pt1_r < img.rows&&pt2_r < img.rows;
		center_r += edge, pt1_r += edge, pt2_r += edge)
	{
		float vec1_y = (float)(pt1_r - center_r);
		float vec2_y = (float)(pt2_r - center_r);

		//myfile << "{" << center_r << "}" << std::endl;

		for (int center_c = startPoint.x + edge / 2, pt1_c = startPoint.x + edge, pt2_c = startPoint.x + edge;
			center_c < img.cols&&pt1_c < img.cols&&pt2_c < img.cols;
			center_c += edge, pt1_c += edge, pt2_c += edge)
		{
			Vec3f vec1 = { (float)(pt1_c - center_c), vec1_y, (float)(img.at<uchar>(pt1_r, pt1_c) - img.at<uchar>(center_r, center_c)) };
			Vec3f vec2 = { (float)(pt2_c - center_c), vec2_y, (float)(img.at<uchar>(pt2_r, pt2_c) - img.at<uchar>(center_r, center_c)) };
			Vec3f	crossProduct = vec1.cross(vec2);
			//myfile <<"["<< crossProduct.val[1]<<"]";
			//myfile << crossProduct;
			//myfile << "[" << atan(crossProduct.val[0] / crossProduct.val[1]) << "]";
			
			GroundMaskFill(Ground.img, Point(center_c, center_r), crossProduct);

		}
		//myfile<< std::endl;
	}
	

	bitwise_and(img, Ground.img, img);
	//imshow("for ground", Ground.img);
	//imshow("depth", img);
	//waitKey();
}


/*parameter:
img : image contains result of cross product
*/
void ObstacleDetection::GroundMaskFill(Mat& img, Point& location, Vec3f& vector)
{
	//from experiment, ground's vector has -ve y coordinate

	/*if (vector.val[0]>0 || vector.val[1]>0 || vector.val[2]>0)
		GroundArrowDraw(img, vector, location);*/


	if ((vector.val[0]>0 || vector.val[2]>0) && vector.val[1]>0)
	{
		float angal = atan(vector.val[0] / vector.val[1]);

#ifdef TEST_SEGMENTATION

		if (angal > minThreashold_horizontalPlane && angal < maxThreashold_horizontalPlane)
		{

			//GroundArrowDraw(img, vector, location);
			GroundMaskUnitFill(img, location);
			//imshow("img", img);
			//waitKey();

		}
		
#else
		//horizontal plane
		if (angal > minThreashold_horizontalPlane && angal < maxThreashold_horizontalPlane)
		{
			int h =(int) GetHeight(location.y, currentRawDepth.at<ushort>(location));
			//myfile << "[" << h << "]";
			if (h < Ground_height && h != -1)
			{
				
				GroundMaskUnitFill(img, location);
				//imshow("img", img);
				//waitKey();
			}
		}
#endif
	}

	
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
	arrowedLine(img, start, end, Scalar(0), 1, 8, 0, 0.5);

}

void ObstacleDetection::GroundMaskUnitFill(Mat& fill, Point& pt)
{
#define edge planeEdgeForPlaneRemove
	// Draw the ground in white
	Point pt1(pt.x - edge / 2, pt.y - edge / 2), pt2(pt.x + edge / 2, pt.y + edge / 2);
	rectangle(fill, pt1, pt2, Scalar(0), CV_FILLED);
}

void ObstacleDetection::createPlaneObject(Mat& src, Mat& img, ObjectType type)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//imshow("Ground.img", Ground.img);
	//waitKey();
	// Find contours of the ground
	findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	vector<Moments> mu(contours.size());
	vector<Point2f> mc(contours.size());
	
	Ground.img=Mat(img.size(), img.type(), Scalar(0));
	Ground.type = type;
	for (size_t i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		mu[i] = moments(contours[i], false);
		mc[i] = Point2f((float)(mu[i].m10 / mu[i].m00), (float)(mu[i].m01 / mu[i].m00));
		//if (area > planeAreaForPlaneRemove&&IsGroundHeight(mc[i],src.at<uchar>(mc[i]))
		if (area > planeAreaForPlaneRemove)
			//save time to draw again since ground img will be used later
			drawContours(Ground.img, contours, i, Scalar(255), CV_FILLED, 8, hierarchy, 0, Point());
			//Ground.Contours.push_back(planeContours[p]);
	}
	
}



void ObstacleDetection::createObstacle(vector<Point> contour, ObjectType type,Point center)
{
	Object obj;
	obj.contour.push_back(contour);
	obj.type = type;
	obj.pos = center;

	ObstacleList.push_back(obj);

}
void ObstacleDetection::createObstacle(vector<Point> contour, ObjectType type, Point center, double area)
{
	Object obj;
	obj.contour.push_back(contour);
	obj.type = type;
	obj.pos = center;
	obj.area = area;
	ObstacleList.push_back(obj);
}
void ObstacleDetection::createObstacle(vector<Point> contour, ObjectType type, double area)
{
	Object obj;
	obj.contour.push_back(contour);
	obj.type = type;
	obj.area = area;
	ObstacleList.push_back(obj);
}


void ObstacleDetection::setCameraAngle(int degree)
{
	CameraAngle = degree;
}

double ObstacleDetection::GetPointAngle(const int pointY)
{
	//myfile << "CameraAngle: " << CameraAngle << std::endl;
	return CameraAngle + (double)(currentDepth.rows / 2.0-(double)pointY) / (double)currentDepth.rows * 43;
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

void ObstacleDetection::SetCurrentRawDepth(Mat* rawDepth) 
{
	currentRawDepth = *rawDepth;
	//GaussianBlur(currentRawDepth, currentRawDepth, Size(1, 13), 0, 0);
}

string ObstacleDetection::findPath()
{

	//histogram with bin width=cols/10, then guassin blur, then find global max, another rect with size row/100 x col/100 repeat the same method
	int width = Ground.img.cols / FirstNumOfBin; // e.g 40 mean total 40bin in the histogram
	int height = Ground.img.rows / 2;
	int area = width*height;
	Mat binImg;
	Mat count = Mat(FirstNumOfBin, 1, CV_32F);
	float sum=0;
	for (int i = 0, j = 0; i < Ground.img.cols; i += width, j++)
	{
		//Point pt1(i, Ground.img.rows), pt2(i + width, height);
		//rectangle(Ground.img, pt1, pt2, Scalar(200), 1);
		binImg = Ground.img(Rect(Point(i, Ground.img.rows), Point(i + width, height)));
		count.at<float>(j) = (float)area - countNonZero(binImg);
		//putText(binImg, std::to_string(count.at<int >(j)), Point(0, 50), FONT_HERSHEY_PLAIN, 0.9, Scalar(128), 1);
		//imshow("binImg", binImg);
		//waitKey();
		sum += count.at<float>(j);
	}
//	OutputDebugStringA(std::to_string(area).c_str());
//	OutputDebugString(L" ");
//	OutputDebugStringA(std::to_string(sum).c_str());

	if (sum < (area*FirstNumOfBin*TooLessGroundPercentage))
		return "no path";

	double max;
	Point maxLoc;
	minMaxLoc(count, NULL, &max, NULL, &maxLoc);

	int FirstPath = maxLoc.y;
	if (FirstPath < FirstNumOfBin / 2)
	{
		for (int i = FirstPath; i < count.rows/2; i++)
		{
			if (std::abs(((float)max - count.at<float>(i))) < ((float)area*0.01))
				FirstPath = i;
		}

	}
	
	//putText(Ground.img, std::to_string(max), Point(FirstPath*width, height), FONT_HERSHEY_PLAIN, 0.9, Scalar(128), 1);
	//circle(Ground.img, Point(FirstPath*width, height), 1, Scalar(128), 3);
	//arrowedLine(Ground.img, Point(FirstPath*width, Ground.img.rows), Point(FirstPath*width, 0), Scalar(100), 2, 8, 0, 0.1);
	//imshow("Ground.img", Ground.img);
	//waitKey();
	//smaller rect


	 int start = FirstPath*width - Ground.img.cols / SecondHistogramRangeColDivisor;
	 if (start < 0)	 start = 0;		 
	 int end = FirstPath*width + Ground.img.cols / SecondHistogramRangeColDivisor;
	 if (end > Ground.img.cols) end = Ground.img.cols;

	 width = (int)((Ground.img.cols / SecondHistogramRangeColDivisor)*2 / SecondNumOfBin); //20bins
	 height = (int)(Ground.img.rows - Ground.img.rows / SecondBinHeightDivisor); //half of the image
	 area = width*height;
	 //OutputDebugStringA(std::to_string(area).c_str());
	// OutputDebugString(L" ");
	 count = Mat(SecondNumOfBin, 1, CV_32F);

	 for (int i = start, j = 0; i < end-width; i += width, j++)
	{
		
		binImg = Ground.img(Rect(Point(i, Ground.img.rows), Point(i + width, height)));
		count.at<float >(j) = (float)area - countNonZero(binImg);
		//Point pt1(i, Ground.img.rows), pt2(i + width, height);
		//rectangle(Ground.img, pt1, pt2, Scalar(128), 1);
		//putText(Ground.img, std::to_string(j), Point(i + width, height), FONT_HERSHEY_PLAIN, 0.9, Scalar(128), 1);
		//imshow("binImg", Ground.img);
		//waitKey();
	}

	minMaxLoc(count, NULL, &max, NULL, &maxLoc);

	//putText(Ground.img, std::to_string(max), Point(maxLoc.y*width + start, height), FONT_HERSHEY_PLAIN, 0.9, Scalar(128), 1);


	int SecondPath = maxLoc.y;
	int beginMax=0;
	int endMax=-1;
		for (int i = 0; i < count.rows; i++)
		{
			if (std::abs(((float)max - count.at<float>(i))) < ((float)area*0.01))
			{
				beginMax = i;
				break;
			}
		}
		//OutputDebugString(L"B's different ");
		//OutputDebugStringA(std::to_string(((float)max - count.at<float>(beginMax))).c_str());
		//OutputDebugString(L" ");
		//	OutputDebugStringA(std::to_string(beginMax).c_str());
		//	OutputDebugString(L"->");
		//	OutputDebugStringA(std::to_string(count.at<float>(beginMax)).c_str());
		//	OutputDebugString(L"B ");

		for (int i = beginMax+1; i < count.rows; i++)
		{
		/*	OutputDebugStringA(std::to_string(i).c_str());
			OutputDebugString(L"->");
			OutputDebugStringA(std::to_string(count.at<float>(i)).c_str());
			OutputDebugString(L" ");*/

			if (std::abs(((float)max - count.at<float>(i))) < ((float)area*0.01))
			{
				endMax = i;
				//OutputDebugStringA(std::to_string(endMax).c_str());
				//OutputDebugString(L"->");
				//OutputDebugString(L"Max ");
			}
		}


		if (beginMax != endMax)
			SecondPath =std::abs(endMax - beginMax) / 2 + beginMax;
		else
			SecondPath = beginMax;

		int finalPath = (SecondPath + 1)*width + start;
		//circle(Ground.img, Point(SecondPath*width + start, height), 1, Scalar(128), 3);
		arrowedLine(Ground.img, Point(finalPath, Ground.img.rows), Point(finalPath, height), Scalar(255), 2, 8, 0, 0.3);
		
		myfile << " " << finalPath << " " << std::endl;
	//imshow("Ground.img", Ground.img);
	//waitKey();

		if (finalPath > 0 && finalPath <= Ground.img.cols / 3)
			return "right";
		if (finalPath > Ground.img.cols / 3 && finalPath <= Ground.img.cols * 2 / 3)
			return "center";
		if (finalPath > Ground.img.cols * 2 / 3 && finalPath < Ground.img.cols)
			return "left";

		return "no path";
}

void ObstacleDetection::Enhance1DMax(Mat *pImg)
{
	Mat temp = Mat(pImg->size(), pImg->type());

	int range = 3; //add 7 bin values to i bin
	for (int i = range; i < pImg->rows-range; i++)
		for (int j = i - range; j <= i + range; j++)
			temp.at<float>(i) += pImg->at<float>(j);
	temp.copyTo(*pImg);
}
