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
	tanList.clear();
	myfile.open(MYFILE_PATH);
	myfile2.open(MYFILE_PATH_h);
};

ObstacleDetection::~ObstacleDetection()
{
	myfile.close();
	myfile2.close();
}

void ObstacleDetection::run(Mat* pImg)
{
	//GaussianBlur(currentRawDepth, currentRawDepth, Size(13, 13), 0, 0);
	//imshow("8bit", *pImg);
	//imshow("raw", currentRawDepth);
	//waitKey(0);
	//GaussianBlur(*pImg, *pImg, Size(13, 13), 0, 0);
	currentDepth =pImg->clone();
	ObstacleList.clear();	
	//std::cout << " Total used : " << t << " seconds" << std::endl;
	GroundMaskCreateRegular(*pImg);	
	Segmentation(*pImg);
	//t = ((double)getTickCount() - t) / getTickFrequency();

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
	//draw conuter to eliminate small segement, the 0th Image is alaways black with no segment and the last one will always the ignore segment
	for (int i = 1; i < numOfSegement - 1; i++)
	{
		obstacleDetect(pThreasholdImageList[i], obstacleMask);
	}
	//myfile << "#ofObstacles: " << ObstacleList.size() << "#ofInterval: " << numOfSegement-1 << std::endl;
#ifdef FOR_REPORT
	imshow("obstacleMask_final", obstacleMask);
	waitKey();
#endif

	//imshow("obstacleMask", obstacleMask);
	//GroundEroAndDilate(Ground.img, GroundBoolMat, dilation_size2,0);
	bitwise_not(Ground.img, Ground.img);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
//	imshow("o", obstacleMask);
	if (!Ground.img.empty())
	{
		
//		imshow("ground", Ground.img);
//		imshow("GroundBoolMatAfterE&D", GroundBoolMat);
		Ground.img &= obstacleMask;
		//findContours(Ground.img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		//for (size_t i = 0; i < contours.size(); i++)
		//	drawContours(Ground.img, contours, i, Scalar(255), -1, 8, hierarchy, 0, Point());
		bitwise_not(Ground.img, Ground.img);
		

#ifdef FOR_REPORT
		imshow("finalGround", Ground.img);
		waitKey();
#endif
		std::string path = findPath();

		if (currentPath.compare(path) != 0)
		{
			currentPath = path;
			//TextToSpeech::pushBack(path);
		}
		if (path.compare("no path") != 0)
			currentDepth &= Ground.img;
	}
	delete[] pThreasholdImageList;
	
	cvtColor(currentDepth, currentDepth, CV_GRAY2RGB);	
	//imwrite(SAMPLE_IMG_PATH + std::to_string(time(NULL)) + ".jpg", currentDepth);
//find path just using ground img
	//imshow("Ground.img",Ground.img);
	//waitKey();

#ifdef DISPLAY_HULL
	vector<Vec4i> hierarchy_HULL;
	for (int i = 0; i < ObstacleList.size(); i++)
		drawContours(currentDepth, ObstacleList[i].contour, 0,
		Scalar(theRNG().uniform(1, 254), theRNG().uniform(1, 254), theRNG().uniform(1, 254)), -1, 8, hierarchy_HULL, 0, Point());
	//imshow("currentDepth", currentDepth);
	//waitKey();
#endif

#ifdef DISPLAY_HEIGHT

	vector<Point> test;
	for (int y = 10; y < currentDepth.rows; y += 24)
	{
		for (int x = 10; x < currentDepth.cols; x += 32)
		{
			test.push_back(Point(x, y));
		}
	}

	for (Point p : test)
	{
		int y = (int)GetHeight(p.y, currentRawDepth.at<ushort>(p));
		putText(currentDepth, std::to_string(y), p, FONT_HERSHEY_PLAIN, 0.7, Scalar(0, 0, 255), 1);
		circle(currentDepth, p, 1, Scalar(0, 0, 255), 3);
	//	putText(currentColor, std::to_string(y), p, FONT_HERSHEY_PLAIN, 1.1, Scalar(0, 0, 255), 1);
	//	circle(currentColor, p, 1, Scalar(0, 0, 255), 3);

	}
#endif


#ifdef DISPLAY_DISTANCE
	vector<Point> test;
	for (int y = 10; y < currentDepth.rows; y += 24)
	{
		for (int x = 10; x < currentDepth.cols; x += 32)
		{
			test.push_back(Point(x, y));
		}
	}

	for (Point p : test)
	{
		//putText(currentDepth, std::to_string(currentRawDepth.at<ushort>(p)), p, FONT_HERSHEY_PLAIN, 0.7, Scalar(0, 0, 255), 1);
		putText(currentDepth, std::to_string(currentDepth.at<uchar>(p)), p, FONT_HERSHEY_PLAIN, 0.7, Scalar(0, 0, 255), 1);
		circle(currentDepth, p, 1, Scalar(0, 0, 255), 3);
	}
#endif	

#ifdef DISPLAY_DIR_LINE

	int	realDepthColsWidth = Ground.img.cols - 20 - 2;
	int realColStart = 20;
	int realColEnd = Ground.img.cols - 2;

	line(currentDepth, Point(realColStart + realDepthColsWidth/3, currentDepth.rows),
		Point(realColStart + realDepthColsWidth / 3, 0),
		Scalar(255), 2, 8, 0);
	line(currentDepth, Point(realColStart + realDepthColsWidth*2 / 3, currentDepth.rows),
		Point(realColStart + realDepthColsWidth * 2 / 3, 0),
		Scalar(255), 2, 8, 0);
	Scalar dir = Scalar(0, 0, 0);
	if (currentPath == "left")
		dir = Scalar(0, 0, 255);
	if (currentPath == "center")
		dir = Scalar(0, 255,0);
	if (currentPath == "right")
		dir = Scalar(255, 0, 0);
	arrowedLine(currentDepth, Point(pathDirCol, currentDepth.rows), Point(pathDirCol, currentDepth.rows-20), dir, 3, 8, 0, 0.6);
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


		mc[i] = Point2f((float)(mu[i].m10 / mu[i].m00), (float)(mu[i].m01 / mu[i].m00));
		createObstacle(hull[i], OBSTACLE, Point((int)mc[i].x, (int)mc[i].y));	

	}
		
}

template<typename T>
void insertAndSort(vector<T> &vec, T newValue)
{
	if (vec.size() == 0)
	{
		vec.push_back(newValue);
		return;
	}
	
	if (vec[0] == newValue)
		return;

	if (vec[0] > newValue)
	{
		vec.insert(vec.begin(), 1, newValue);
		return;
	}

	for (int i = 1; i < vec.size(); i++)
	{
		if (vec[i] == newValue)
			return;
		if (vec[i - 1]<newValue&&vec[i]>newValue)
		{
			vec.insert(vec.begin() + i, 1, newValue);
			return;
		}
			
	}

	vec.push_back(newValue);


}

int ObstacleDetection::Rand()
{
	srand(time(NULL));
	return std::rand() % planeAreaForPlaneRemove + 1;
}

Point ObstacleDetection::RandPoint(Point start)
{
	
	return start;
}


Vec3f ObstacleDetection::RandVector(Vec3f &vec1, Vec3f &vec2)
{
	Point P1, P2, P3, P4;


	return vec1;
}
void ObstacleDetection::GroundMaskCreateRandom(Mat &img)
{
	int m_edge = planeEdgeForPlaneRemove;

	//(col,row)=(x,y)
	Point startPoint = Point(0, img.rows / 2 - 1);
	//myfile << "img.rows: " << img.rows << std::endl;
	//Point startPoint = Point(0, m_edge-1);
	Ground.img = Mat(img.size(), CV_8UC1, Scalar(255));
	Mat whitePaper = Mat(img.rows * 4, img.cols * 4, CV_8UC1, Scalar(255));
	putText(whitePaper, std::to_string(CameraAngle), Point(50, 50), FONT_HERSHEY_PLAIN, 0.9, Scalar(128), 1);
	vector<float> tanList;
#ifdef FOR_REPORT
	Mat temp = Mat(img.size(), CV_8UC1, Scalar(255));
#endif

	for (int center_r = startPoint.y + m_edge / 2, pt1_r = startPoint.y, pt2_r = startPoint.y + m_edge, interval_r = 0;
		center_r < img.rows&&pt1_r < img.rows&&pt2_r < img.rows;
		center_r += m_edge, pt1_r += m_edge, pt2_r += m_edge, interval_r += m_edge)
	{
		float vec1_y = (float)(pt1_r - center_r);
		float vec2_y = (float)(pt2_r - center_r); 

		//myfile << "{" << center_r << "}" << std::endl;

		for (int center_c = startPoint.x + m_edge / 2, pt1_c = startPoint.x + m_edge, pt2_c = startPoint.x + m_edge, interval_c = 0;
			center_c < img.cols&&pt1_c < img.cols&&pt2_c < img.cols;
			center_c += m_edge, pt1_c += m_edge, pt2_c += m_edge, interval_c += m_edge)
		{
			Vec3f vec1 = { (float)(pt1_c - center_c), vec1_y, (float)(img.at<uchar>(pt1_r, pt1_c) - img.at<uchar>(center_r, center_c)) };
			Vec3f vec2 = { (float)(pt2_c - center_c), vec2_y, (float)(img.at<uchar>(pt2_r, pt2_c) - img.at<uchar>(center_r, center_c)) };
			Vec3f	crossProduct = vec1.cross(vec2);
			//myfile <<"["<< crossProduct.val[1]<<"]";
			//myfile << crossProduct;
			//myfile << "[" << atan(crossProduct.val[0] / crossProduct.val[1]) << "]";

			//GroundMaskFill(Ground.img, Point(center_c, center_r), crossProduct);
			GroundArrowDrawOnWhitePaper(whitePaper, crossProduct, Point(center_c + interval_c * 2, center_r + interval_r * 2));

			if (crossProduct[0] > 0 || crossProduct[1]>0)
			{
				float tan = atan(crossProduct.val[0] / crossProduct.val[1]);
				insertAndSort<float>(tanList, tan);
			}
		}
		//myfile<< std::endl;
	}
	//GroundDefault(Ground.img);
	for (int i = 0; i < tanList.size(); i++)
	{
		myfile << tanList[i] << " ";
	}
	myfile << std::endl;
	//StairDetection stairs;
	//std::vector<cv::Point> stairConvexHull;
	//std::vector<std::vector<cv::Point> > hull(1);
	//stairs.Run(currentColor, currentDepth, Ground.img, stairConvexHull);
	//if (!stairConvexHull.empty()) {
	//	cv::Mat temp = currentColor.clone();
	//	cv::Scalar color = Scalar(cv::theRNG().uniform(0, 255), cv::theRNG().uniform(0, 255), cv::theRNG().uniform(0, 255));
	//	hull.push_back(stairConvexHull);
	//	for (int i = 0; i<hull.size(); ++i) {
	//		drawContours(temp, hull, i, color, 3, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
	//	}
	//	//imshow(std::to_string(cv::getTickCount()), temp);
	//	//imshow("s", temp);
	//}

#ifdef FOR_REPORT
	Ground.img.copyTo(temp);
#endif
#ifdef FOR_REPORT
	imshow("original_depth", img);
	waitKey();
#endif

	//bitwise_and(img, Ground.img, img);

#ifdef FOR_REPORT
	imshow("ground", temp);
	imshow("brief ground in srcDepth", img);
	waitKey();
#endif


	//imshow("for ground", Ground.img);
	imshow("whitePaper", whitePaper);
	//imshow("depth", img);
	waitKey(1);
}

float ObstacleDetection::GroundDirection(Vec3f& vector)
{
	//return atan(vector.val[0] / vector.val[1]);
	return (float)atan(sqrt(vector.val[0] * vector.val[0] + vector.val[2] * vector.val[2])/abs(vector.val[1])  );
}

void ObstacleDetection::GroundMaskCreateRegular(Mat &img)
{
	
	int m_edge = planeEdgeForPlaneRemove;

	//(col,row)=(x,y)
	Point startPoint = Point(0, img.rows / 2 - 1);
	//myfile << "img.rows: " << img.rows << std::endl;
	//Point startPoint = Point(0, m_edge-1);
	Ground.img = Mat(img.size(), CV_8UC1,Scalar(255));
	GroundBoolMat = Mat(img.rows / m_edge, img.cols / m_edge, CV_8UC1, Scalar(0));
	Mat whitePaper = Mat(img.rows * 4, img.cols * 4, CV_8UC1, Scalar(255));
	putText(whitePaper, std::to_string(CameraAngle), Point(50, 50), FONT_HERSHEY_PLAIN, 0.9, Scalar(128), 1);

#ifdef FOR_REPORT
	Mat temp = Mat(img.size(), CV_8UC1, Scalar(255));
#endif

	for (int center_r = startPoint.y + m_edge / 2, pt1_r = startPoint.y, pt2_r = startPoint.y + m_edge, interval_r=0;
		center_r < img.rows&&pt1_r < img.rows&&pt2_r < img.rows;
		center_r += m_edge, pt1_r += m_edge, pt2_r += m_edge, interval_r += m_edge)
	{
		float vec1_y = (float)(pt1_r - center_r);
		float vec2_y = (float)(pt2_r - center_r);

		//myfile << "{" << center_r << "}" << std::endl;

		for (int center_c = startPoint.x + m_edge / 2, pt1_c = startPoint.x + m_edge, pt2_c = startPoint.x + m_edge,interval_c=0;
			center_c < img.cols&&pt1_c < img.cols&&pt2_c < img.cols;
			center_c += m_edge, pt1_c += m_edge, pt2_c += m_edge, interval_c += m_edge)
		{
			Vec3f vec1 = { (float)(pt1_c - center_c), vec1_y, (float)(currentRawDepth.at<ushort>(pt1_r, pt1_c) - currentRawDepth.at<ushort>(center_r, center_c)) };
			Vec3f vec2 = { (float)(pt2_c - center_c), vec2_y, (float)(currentRawDepth.at<ushort>(pt2_r, pt2_c) - currentRawDepth.at<ushort>(center_r, center_c)) };
			//Vec3f vec1 = { (float)(pt1_c - center_c), vec1_y, (float)(img.at<uchar>(pt1_r, pt1_c) - img.at<uchar>(center_r, center_c)) };
			//Vec3f vec2 = { (float)(pt2_c - center_c), vec2_y, (float)(img.at<uchar>(pt2_r, pt2_c) - img.at<uchar>(center_r, center_c)) };
			Vec3f	crossProduct = vec2.cross(vec1);

			//myfile <<"["<< crossProduct.val[1]<<"]";
			//myfile << crossProduct << ";";
			//myfile << "[" << GroundDirection(crossProduct) << "]";
			
			GroundMaskFill(Ground.img, GroundBoolMat, Point(center_c, center_r), crossProduct);
			
#ifdef DISPLAY_ARROW
			GroundArrowDrawOnWhitePaper(whitePaper, crossProduct, Point(center_c + interval_c * 2, center_r + interval_r*2));
#endif
#ifdef COLLECT_TAN_LIST
			if (crossProduct[0] > 0 || crossProduct[1]>0 )
			{
				float tan = GroundDirection(crossProduct);
				myfile << GroundDirection(crossProduct) << std::endl;
				//insertAndSort<float>(tanList, tan);
			}
#endif
		}

		//myfile<< std::endl;
	}
	//GroundDefault(Ground.img);

#ifdef COLLECT_TAN_LIST
	//for (int i = 0; i < tanList.size(); i++)
	//{
	//	myfile << tanList[i] << std::endl;
	//}
	//myfile <<"==========================="<< std::endl;
#endif
	
#ifdef FOR_REPORT
	Ground.img.copyTo(temp);
#endif
#ifdef FOR_REPORT
	imshow("original_depth", img);
	waitKey();
#endif
	GroundEroAndDilate(Ground.img, GroundBoolMat, dilation_size1, erosion_size);
	//GroundRefine(GroundBoolMat, Ground.img);
	bitwise_and(img, Ground.img, img);

#ifdef FOR_REPORT
	imshow("ground", temp);
	imshow("brief ground in srcDepth", img);
	waitKey();
#endif

	


	//imshow("for ground", Ground.img);
#ifdef DISPLAY_ARROW
	imshow("whitePaper", whitePaper);
#endif
	//imshow("GroundBoolMat", GroundBoolMat);
	//imshow("depth", img);
	//waitKey(1);
}

void ObstacleDetection::GroundDefault(Mat& img)
{
	Point pt1(0, img.rows), pt2(img.cols, img.rows - planeEdgeForPlaneRemove);
	rectangle(img, pt1, pt2, Scalar(0), CV_FILLED);
}

void ObstacleDetection::GroundEroAndDilate(Mat& img, Mat& boolMat, int diSize,int eroSize)
{
	int erosion_type;
	if (erosion_elem == 0){ erosion_type = MORPH_RECT; }
	else if (erosion_elem == 1){ erosion_type = MORPH_CROSS; }
	else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

	Mat elementE = getStructuringElement(erosion_type,
		Size(2 * eroSize + 1, 2 * eroSize + 1),
		Point(eroSize, eroSize));

	/// Apply the erosion operation
	//erode(boolMat, boolMat, elementE);


	int dilation_type;
	if (dilation_elem == 0){ dilation_type = MORPH_RECT; }
	else if (dilation_elem == 1){ dilation_type = MORPH_CROSS; }
	else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

	Mat elementD = getStructuringElement(dilation_type,
		Size(2 * diSize + 1, 2 * diSize + 1),
		Point(diSize, diSize));
	/// Apply the dilation operation
	dilate(boolMat, boolMat, elementD);

	GroundRefine(boolMat, img);
}
void ObstacleDetection::GroundRefine(Mat& boolMat, Mat& img)
{
	for (int i = 0; i < boolMat.rows; i++)
	{
		for (int j = 0; j < boolMat.cols; j++)
			if (boolMat.at<uchar>(i, j) == 255)
			{
				Point location((j*planeEdgeForPlaneRemove) + (planeEdgeForPlaneRemove / 2), (i * planeEdgeForPlaneRemove) + (planeEdgeForPlaneRemove-1) + (planeEdgeForPlaneRemove / 2));
	//			myfile << "[" << j << "," << i << "]=" ;
		//		myfile << location << " ";
				GroundMaskUnitFill(img, location);
			}	
		//myfile << std::endl;
	}
	
}

/*parameter:
img : image contains result of cross product
*/
void ObstacleDetection::GroundMaskFill(Mat& img,Mat& boolMat, Point& location, Vec3f& vector)
{
	//from experiment, ground's vector has -ve y coordinate

	//if (vector.val[0]>0 || vector.val[1]>0 || vector.val[2]>0)
		//GroundArrowDraw(img, vector, location);


	if ((vector.val[0]!=0 || vector.val[2]!=0) && vector.val[1]<0)
	{
		float angal = GroundDirection(vector);
		
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
			//myfile2 << h << std::endl;
			if (h < Ground_height && h != -1)
			{
				//myfile << "<-G";
		//		GroundMaskUnitFill(img, location);
				
				//std::cout << location;
				//imshow("boolMat", boolMat);
				//imshow("img", img);
				//waitKey();
			//	myfile << location << " ";
				GroundBoolMatUnitFill(boolMat, location);
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
	arrowedLine(img, start, end, Scalar(255), 1, 8, 0, 0.5);
}

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


void ObstacleDetection::GroundBoolMatUnitFill(Mat& fill, Point& pt)
{
	Point pt1(pt.x - planeEdgeForPlaneRemove / 2, pt.y - planeEdgeForPlaneRemove / 2);
	
	fill.at<uchar>(pt1.y / planeEdgeForPlaneRemove,pt1.x / planeEdgeForPlaneRemove ) = 255;
	//std::cout << "to" << pt1 << " ";
}

void ObstacleDetection::GroundMaskUnitFill(Mat& fill, Point& pt)
{
	// Draw the ground in white
	Point pt1(pt.x - planeEdgeForPlaneRemove / 2, pt.y - planeEdgeForPlaneRemove / 2), pt2(pt.x + planeEdgeForPlaneRemove / 2, pt.y + planeEdgeForPlaneRemove / 2);
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

int ObstacleDetection::getCameraAngle()
{
	return CameraAngle;
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
	//invalid width [0:45], [img.col-5,img.col]
	//histogram with bin width=cols/10, then guassin blur, then find global max, another rect with size row/100 x col/100 repeat the same method
	int realDepthColsWidth ;
	int realColStart ;
	int realColEnd ;

	if (Ground.img.cols == 320)
	{
		realDepthColsWidth = Ground.img.cols - 20-2;
		realColStart = 20;
		realColEnd = Ground.img.cols-2;
	}
	
	if (Ground.img.cols == 640)
	{
		 realDepthColsWidth = Ground.img.cols - 45 - 5;
		 realColStart = 45;
		 realColEnd = Ground.img.cols - 5;
	}

	int width = realDepthColsWidth / FirstNumOfBin; // e.g 40 mean total 40bin in the histogram
	int height = Ground.img.rows / 2;
	int area = width*height;
	Mat binImg;
	Mat count = Mat(FirstNumOfBin, 1, CV_32F);
	float sum=0;
	//myfile << "lenght:" <<  realDepthColsWidth << " |";
	for (int i = 0; i < FirstNumOfBin; i++)
	{
		binImg = Ground.img(Rect(Point(realColStart + width*i, Ground.img.rows), Point(realColStart + width*i + width, height)));
		count.at<float>(i) = (float)area - countNonZero(binImg);
		//Point pt1(realColStart + width*i, Ground.img.rows), pt2(realColStart + width*i + width, height);
		//rectangle(Ground.img, pt1, pt2, Scalar(200), 1);
		
		//putText(binImg, std::to_string(count.at<int >(j)), Point(0, 50), FONT_HERSHEY_PLAIN, 0.9, Scalar(128), 1);
		//imshow("binImg", binImg);
		//waitKey();
		//myfile <<"[" <<i << "]" << count.at<float>(i)<<" ";
		sum += count.at<float>(i);
	}
	//myfile << std::endl;
//	OutputDebugStringA(std::to_string(area).c_str());
//	OutputDebugString(L" ");
//	OutputDebugStringA(std::to_string(sum).c_str());
//	myfile << "first path" << std::endl;
	if (sum < (planeAreaForPlaneRemove*(dilation_size1 * 2 + 1)*(dilation_size1 * 2 + 1)))
	{
#ifdef record_noGround
		time_t timer;
		imwrite("C:/Users/HOHO/Pictures/sample/result/noGround/" + std::to_string(time(&timer)) + ".bmp",currentDepth);
#endif
		return "no path";
	}

	double max;
	Point maxLoc;
	minMaxLoc(count, NULL, &max, NULL, &maxLoc);

	int FirstPath = maxLoc.y;
	if (FirstPath < FirstNumOfBin / 2)
	{
		for (int i = FirstPath; i < count.rows/2; i++)
		{
			if (std::abs(((float)max - count.at<float>(i))) < ((float)planeEdgeForPlaneRemove*width))
				FirstPath = i;
		}

	}
	//myfile << "FirstPath:" << FirstPath << std::endl;
	//putText(Ground.img, std::to_string(max), Point(FirstPath*width, height), FONT_HERSHEY_PLAIN, 0.9, Scalar(128), 1);
	//circle(Ground.img, Point(FirstPath*width, height), 1, Scalar(128), 3);
	//arrowedLine(Ground.img, Point(realColStart + FirstPath*width, Ground.img.rows), Point(realColStart+FirstPath*width, 0), Scalar(100), 2, 8, 0, 0.1);
	//imshow("Ground.img", Ground.img);
	//waitKey();
	//smaller rect


	int start = realColStart+FirstPath*width - humanShoulderLength / 2;
	 if (start < realColStart)	 start = realColStart;
	 int end = start + humanShoulderLength;
	 if (end > realColEnd) end = realColEnd;

	 width = (int)(humanShoulderLength / SecondNumOfBin); //20bins
	 height = (int) (humanShoulderLength/2); //half of the image
	 area = width*height;
	 //OutputDebugStringA(std::to_string(area).c_str());
	// OutputDebugString(L" ");
	 count = Mat(SecondNumOfBin, 1, CV_32F);
	 //myfile << "start:" << start << std::endl;
	 for (int i = 0; i < SecondNumOfBin&&(start + i*width + width)<end; i++)
	{
		//myfile << "[" << i << "]" << Point(start + i*width, Ground.img.rows) << " " << Point(start + i*width + width, Ground.img.rows - height) << std::endl;
		binImg = Ground.img(Rect(Point(start + i*width, Ground.img.rows), Point(start + i*width + width, Ground.img.rows - height)));
		count.at<float >(i) = (float)area - countNonZero(binImg);
		//Point pt1(start + i*width, Ground.img.rows), pt2(start + i*width + width, Ground.img.rows - height);
		//rectangle(Ground.img, pt1, pt2, Scalar(200), 1);
		//putText(Ground.img, std::to_string(j), Point(i + width, height), FONT_HERSHEY_PLAIN, 0.9, Scalar(128), 1);
		//imshow("binImg", Ground.img);
		//waitKey();
		//myfile << "[" << i << "]" << count.at<float >(i) << std::endl;
	}
	// myfile << "second path" << std::endl;
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
		//myfile << "finalPath" << SecondPath << std::endl;
		int finalPath = ((float)SecondPath + 0.5)*(float)width + start;
		//circle(Ground.img, Point(SecondPath*width + start, height), 1, Scalar(128), 3);
		
		
		//myfile << " " << finalPath << " " << std::endl;
	//imshow("Ground.img", Ground.img);
	//waitKey();
		pathDirCol = finalPath;
		if ((finalPath > 0) && (finalPath <= realColStart+realDepthColsWidth / 3))
			return "right";
		if ((finalPath > realColStart + realDepthColsWidth / 3) && (finalPath <= realColStart+realDepthColsWidth * 2 / 3))
			return "center";
		if ((finalPath > realColStart + realDepthColsWidth * 2 / 3) && (finalPath < realColStart+realDepthColsWidth))
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

void ObstacleDetection::testResult(int keyInput, int timeStamp)
{
	if (keyInput ==(int) 'n')
		myfile << "correct " << timeStamp<<std::endl;
	else if (keyInput == (int)'m')
		myfile << "over " << timeStamp<< std::endl;
	else
		myfile << "incorrect " << timeStamp<<std::endl;
}