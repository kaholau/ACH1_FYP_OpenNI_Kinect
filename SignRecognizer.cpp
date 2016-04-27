/* ----------------------------------------------------------------------------
** SignRecognizer.cpp
**
** Date: 2016-01-13
** Description: This is the header file of class 'SignRecognizer'. This class
** is currently for detecting the floor signs in HKUST. More and different
** signs may be recognized later.
** This class requires the use of Microsoft Speech API(SAPI) and Tesseract OCR.
** Please install them and include the path and library first.
**
** Microsoft Speech API (SAPI) 5.3 Reference:
** https://msdn.microsoft.com/en-us/library/ms720161(v=vs.85).aspx
** https://msdn.microsoft.com/en-us/library/ms720163(v=vs.85).aspx
** Microsoft Speech SDK 5.1 Download:
** https://www.microsoft.com/en-us/download/details.aspx?id=10121
**
** Tesseract OCR Information and Download:
** https://github.com/tesseract-ocr
**
**
** Author: Chan Tong Yan, Lau Ka Ho, Joel @ HKUST
** E-mail: yumichanhk2014@gmail.com
**
** --------------------------------------------------------------------------*/
#include "SignRecognizer.h"
#include <future>

using namespace cv;


int currSet = 0;


const std::wstring SignRecognizer::STRING_UPPER_FLOOR[8] =
{
	L"Ground floor",
	L"first floor",
	L"second floor",
	L"third floor",
	L"fourth floor",
	L"fifth floor",
	L"sixth floor",
	L"seventh floor"
};
const std::wstring SignRecognizer::STRING_LG1 = L"LG1";
const std::wstring SignRecognizer::STRING_LG2 = L"LG2";
const std::wstring SignRecognizer::STRING_LG3 = L"LG3";
const std::wstring SignRecognizer::STRING_LG4 = L"LG4";
const std::wstring SignRecognizer::STRING_LG5 = L"LG5";
const std::wstring SignRecognizer::STRING_LG6 = L"LG6";
const std::wstring SignRecognizer::STRING_LG7 = L"LG7";

SignRecognizer::SignRecognizer()
{
	frameSize.width = DEFAULT_FRAME_WIDTH;
	frameSize.height = DEFAULT_FRAME_HEIGHT;

	isLaterallyInverted = true;
	scale = 0.0;
	lowThreshold = 45;
	dilateSize = 1;

	map_x.create(frameSize, CV_32FC1);
	map_y.create(frameSize, CV_32FC1);
	for (int i = 0; i < frameSize.height; i++)
	{
		for (int j = 0; j < frameSize.width; j++)
		{
			map_x.at<float>(i, j) = (float)(frameSize.width - j);
			map_y.at<float>(i, j) = (float)(i);
		}
	}

	tess.Init(0, "eng", tesseract::OEM_DEFAULT);
	tess.SetPageSegMode(tesseract::PSM_SINGLE_WORD);
	tess.SetVariable("tessedit_char_whitelist", "LG1234567");
}

SignRecognizer::SignRecognizer(int w = DEFAULT_FRAME_WIDTH, int h = DEFAULT_FRAME_HEIGHT, bool isLateralInvert = true)
{
	frameSize.width = w;
	frameSize.height = h;

	isLaterallyInverted = isLateralInvert;
	scale = 0.0;
	lowThreshold = 45;
	dilateSize = 2;
	
	if (isLaterallyInverted)
	{
		map_x.create(frameSize, CV_32FC1);
		map_y.create(frameSize, CV_32FC1);
		for (int i = 0; i < frameSize.height; i++)
		{
			for (int j = 0; j < frameSize.width; j++)
			{
				map_x.at<float>(i, j) = (float)(frameSize.width - j);
				map_y.at<float>(i, j) = (float)(i);
			}
		}
	}

	tess.Init(0, "eng", tesseract::OEM_DEFAULT);
	tess.SetPageSegMode(tesseract::PSM_SINGLE_WORD);
	tess.SetVariable("tessedit_char_whitelist", "LG1234567");
}


SignRecognizer::~SignRecognizer()
{
}

void SignRecognizer::runRecognizer(cv::Mat &frame, std::string fName)
{
#ifdef DURATION_CHECK
	double time = 0;
	uint64_t oldCount = 0, curCount = 0;
	curCount = cv::getTickCount();
#endif

	std::ostringstream sout;
	int k = 0;
	cv::Mat frameGray;
	std::vector<std::vector<cv::Point>> contours_all;
	std::vector<cv::Vec4i> hierarchy;
	getContoursOfFrame(frame, frameGray, contours_all, hierarchy); 

	double a, b;
	double area;
	double avg_tmp = 0;
	double distance;
	cv::Mat *sign = NULL;
	cv::Mat *smallSign = NULL;
	cv::Rect rect;
#ifdef SHOW_IMAGE_AND_RESULT
	std::vector<cv::Mat> savedSigns;
#endif
	std::vector<int> savedSigns_index;
	std::vector<int>::iterator it_end_index;
	for (k = (int)contours_all.size() - 1; k >= 0; --k)
	{
		area = contourArea(contours_all[k]);
		//std::cout << "Area = " << area << ",\t";
		if ((area < CONTOUR_AREA_THRESHOLD) || (area > CONTOUR_AREA_MAX_THRESHOLD))
			continue;


		// Calculate the similarity (confidence? distance?) between the current
		// contour and the previous contour saved
		if (savedSigns_index.size() > 0)
		{
			it_end_index = savedSigns_index.end();
			it_end_index--;

			distance = matchShapes(contours_all[k], contours_all[*it_end_index],
				CV_CONTOURS_MATCH_I3, 0);
#ifdef SHOW_DEBUG_MESSAGES
			std::cout << "Contour[" << k << "] VS last SavedContour, distance = " << distance;
#endif
			if (distance < SAME_CONTOUR_THRESHOLD)
			{
#ifdef SHOW_DEBUG_MESSAGES
				std::cout << " => same\n";
				OutputDebugStringA(" => same\n");
#endif
				continue;
			}
		}
#ifdef SHOW_DEBUG_MESSAGES
		std::cout << std::endl;
#endif

		// Draw the contours first in order to obtain a matrix of each contour
		Mat contourImg = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
		drawContours(contourImg, contours_all, k, Scalar::all(MAX_VALUE_8BITS), 
			CV_FILLED, 8, hierarchy, 0, Point());

		// Find the best-fit ellipse by conic fitting
		RotatedRect rrect = fitEllipse(contours_all[k]);
		Mat foundEllipse(frameGray.size(), frameGray.type(), Scalar::all(0));
		rrect.size.width *= 0.99;
		rrect.size.height *= 0.99;
		ellipse(foundEllipse, rrect, Scalar::all(MAX_VALUE_8BITS), CV_FILLED);

		// After an ellipse is found, cut and save it
		rect.width = (int)(rrect.size.width);
		rect.height = (int)(rrect.size.height);
		rect.x = (int)(rrect.center.x - rrect.size.width / 2);
		rect.y = (int)(rrect.center.y - rrect.size.height / 2);
		if (rect.x < 0 || rect.y < 0 || rect.width < 0 || rect.height < 0 ||
			(rect.x + rect.width) > foundEllipse.cols || (rect.y + rect.height) > foundEllipse.rows)
			continue;

		cv::Mat ellipseMask(foundEllipse, rect);

		a = sqrt(contourImg.dot(contourImg));
		b = sqrt(foundEllipse.dot(foundEllipse));
		distance = (foundEllipse.dot(contourImg)) / a / b;
#ifdef SHOW_DEBUG_MESSAGES
		std::cout << "SavedContour[" << k << "] & estimated ellipse, distance = " << distance;
#endif

		// Determine whether there is any contour of ellipse
		if (distance > SAME_ELLIPSE_THRESHOLD) { // if there is an ellipse
#ifdef SHOW_DEBUG_MESSAGES
			std::cout << " => Ellipse Found\n";
			OutputDebugStringA(" => Ellipse Found\n");
#endif

#ifdef SHOW_MARKERS
			ellipse(frame, rrect, Scalar(0, MAX_VALUE_8BITS / 2, MAX_VALUE_8BITS), 3); // Draw the ellipse on the image
#endif

			// Binarize the sign
			sign = new Mat(frameGray(rect) & ellipseMask);
#ifdef SAVE_IMAGE_AND_RESULT
			sout.str("");
			sout << fName << "_sign_original" << k << ".bmp";
			cv::imwrite(sout.str(), *sign);
#endif

			avg_tmp = cv::mean(*sign, ellipseMask)[0];
			avg_tmp *= AVG_GRAYSCALE_SCALE;

#ifdef SHOW_DEBUG_MESSAGES
			std::cout << "Grayscale average: " << avg_tmp << std::endl;
#endif

			cv::Mat ellipseMask_inv(ellipseMask.size(), ellipseMask.type(), Scalar::all(0));
			ellipseMask_inv = MAX_VALUE_8BITS - ellipseMask;
			//cv::bitwise_not(ellipseMask, ellipseMask_inv);
			cv::threshold(*sign, *sign, avg_tmp, ABSOLUTE_WHITE, THRESH_BINARY);

#ifdef SAVE_IMAGE_AND_RESULT
			sout.str("");
			sout << fName << "_sign_orig" << k << ".bmp";
			cv::imwrite(sout.str(), *sign);
#endif

			*sign |= ellipseMask_inv;
			floodFill(*sign, Point(0, 0), Scalar::all(0));
			floodFill(*sign, Point(sign->cols - 1, 0), Scalar::all(0));
			floodFill(*sign, Point(0, sign->rows - 1), Scalar::all(0));
			floodFill(*sign, Point(sign->cols - 1, sign->rows - 1), Scalar::all(0));

			if (sign->cols > SIGN_WIDTH_THRESHOLD) {
				scale = (double)SIGN_WIDTH_THRESHOLD / (double)sign->cols;
				Size sSize((int)(scale*sign->size().width), (int)(scale*sign->size().height));
				smallSign = new cv::Mat(sSize, sign->type(), Scalar::all(0));
				cv::resize(*sign, *smallSign, sSize);
				delete sign;
				sign = smallSign;
				smallSign = NULL;
			}

			// Erosion
			Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(1, 1));
			erode(*sign, *sign, element);
#ifdef SAVE_IMAGE_AND_RESULT
			sout.str("");
			sout << fName << "_sign_" << k << "_ero.bmp";
			cv::imwrite(sout.str(), *sign);
#endif


			//medianBlur(*sign, *sign, MEDIAN_BLUR_KSIZE);
			if (isLaterallyInverted)
			{
				sign = invertLaterally(sign);

//#ifdef SAVE_IMAGE_AND_RESULT
//				sout.str("");
//				sout << fName << "_sign_" << k << ".bmp";
//				cv::imwrite(sout.str(), *sign);
//#endif
			}

			// apply OCR to obtain the characters
			tess.SetImage((uchar*)(sign->data), sign->cols, sign->rows, 1, sign->step1());
			string text = tess.GetUTF8Text();
			text.erase(remove(text.begin(), text.end(), '\n'), text.end());
			text.erase(remove(text.begin(), text.end(), ' '), text.end());

			//if ((strcmp(text.c_str(), "") == 0) || (strcmp(lastDet.c_str(), text.c_str()) == 0))
			//	continue;

#ifdef SHOW_DEBUG_MESSAGES
			std::cout << "SavedContour[" << k << "] text: " << text << "_Floor";
#endif
			std::cout << "SavedContour[" << k << "] text: " << text << "_Floor";
			std::wstring out;
			if (!getResultString(text, out)) {
				std::cerr << std::endl << "Error on getResultString()" << std::endl;
				continue;
			}
			TextToSpeech::pushBack(out);

			cv::putText(frame, text, cv::Point(contours_all[k][1].x, contours_all[k][1].y-10), cv::FONT_HERSHEY_SIMPLEX, 4,
				cv::Scalar(0, 128, 255), 8);
			savedSigns_index.push_back(k);

#ifdef TEST_SIGN
			fout << fName << "," << text << "," << out.c_str() << "," << ((out == L"LG3")?1:0) << std::endl;
#endif

#ifdef SHOW_DEBUG_MESSAGES
			std::cout << std::endl;
#endif

#ifdef SHOW_IMAGE_AND_RESULT
			// Save and show the sign
			savedSigns.push_back(*sign);

			sout.str("");
			sout << fName << "_sign_" << k << ".bmp";
			//namedWindow(sout.str(), CV_WINDOW_AUTOSIZE);
			//cv::imshow(sout.str(), *sign);

#ifdef SAVE_IMAGE_AND_RESULT
			cv::imwrite(sout.str(), *sign);
#endif
#endif

#ifdef SHOW_IMAGE_AND_RESULT
			sout.str("");
			sout << fName << "_Contours_" << k << IMAGE_EXTENSION;
			cv::imshow(sout.str(), contourImg);
			cv::imwrite(sout.str(), contourImg);
			waitKey(10);
#endif

			lastDet = text;
			sign = NULL;
		}
	}
#ifdef SHOW_DEBUG_MESSAGES
	std::cout << std::endl << "No of Contours saved = " << savedSigns_index.size() << std::endl;
#endif

	//resize(frame, frame, cv::Size(480, 360));

#ifdef SHOW_IMAGE_AND_RESULT
#ifdef SAVE_IMAGE_AND_RESULT
	// Show and save the result
	sout.str("");
	sout << fName << "_result.bmp";
	cv::imwrite(sout.str(), frame);
#endif

	namedWindow("Detection Result", CV_WINDOW_NORMAL);
	cv::imshow("Detection Result", frame);
#endif

#ifdef DURATION_CHECK
	time = (cv::getTickCount() - curCount) / cv::getTickFrequency();
	printf("SignRec Duration: %f\n", time);
#endif

	return;
}


void SignRecognizer::setFrameSize(int w, int h)
{
	if (frameSize.width != w)
		frameSize.width = w;

	if (frameSize.height != h)
		frameSize.height = h;

	updateMap();

	return;
}

bool SignRecognizer::getResultString(std::string &in, std::wstring &out)
{
	if (in.length() == 1)
	{
		char c = *(in.c_str());
		std::cout << "\t\t found: " << c << std::endl;
		switch (c)
		{
		case 'G':
			out = std::wstring(STRING_UPPER_FLOOR[0]);
			return true;
			break;

		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
			out = std::wstring(STRING_UPPER_FLOOR[(int)c - 48]);
			return true;
			break;

		default:
			return false;
			break;
		}
	}

	/* if (strcmp(in.c_str(), "1") == 0)
	{
		out = std::wstring(STRING_FIRST_FLOOR);
	}*/
	if ((strcmp(in.c_str(), "LG7") == 0) || (strcmp(in.c_str(), "G7") == 0)
		|| (strcmp(in.c_str(), "67") == 0) || (strcmp(in.c_str(), "L67") == 0)
		|| (strcmp(in.c_str(), "L7") == 0) || (strcmp(in.c_str(), "767") == 0))
	{
		out = std::wstring(L"LG7");
	}
	else if ((strcmp(in.c_str(), "LG1") == 0) || (strcmp(in.c_str(), "G1") == 0)
		|| (strcmp(in.c_str(), "61") == 0) || (strcmp(in.c_str(), "L61") == 0)
		|| (strcmp(in.c_str(), "L1") == 0) || (strcmp(in.c_str(), "761") == 0))
	{
		out = std::wstring(L"LG1");
	}
	else if ((strcmp(in.c_str(), "LG5") == 0) || (strcmp(in.c_str(), "G5") == 0)
		|| (strcmp(in.c_str(), "65") == 0) || (strcmp(in.c_str(), "L65") == 0)
		|| (strcmp(in.c_str(), "L5") == 0) || (strcmp(in.c_str(), "765") == 0))
	{
		out = std::wstring(L"LG5");
	}
	else if ((strcmp(in.c_str(), "LG3") == 0) || (strcmp(in.c_str(), "G3") == 0)
		|| (strcmp(in.c_str(), "63") == 0) || (strcmp(in.c_str(), "L63") == 0)
		|| (strcmp(in.c_str(), "L3") == 0) || (strcmp(in.c_str(), "763") == 0))
	{
		out = std::wstring(L"LG3");
	}
	else if ((strcmp(in.c_str(), "LG2") == 0) || (strcmp(in.c_str(), "G2") == 0)
		|| (strcmp(in.c_str(), "62") == 0) || (strcmp(in.c_str(), "L62") == 0)
		|| (strcmp(in.c_str(), "L2") == 0) || (strcmp(in.c_str(), "762") == 0))
	{
		out = std::wstring(L"LG7");
	}
	else
	{
		out = std::wstring(in.begin(), in.end());
		return false;
	}

	return true;
}

void SignRecognizer::getContoursOfFrame(cv::Mat &frame, cv::Mat &grayOut, std::vector<std::vector<Point>> &contours, std::vector<Vec4i> &hierarchy)
{
#ifdef DURATION_CHECK
	double time = 0;
	uint64_t oldCount = 0, curCount = 0;
	curCount = cv::getTickCount();
#endif

	//if (isLaterallyInverted)
	//{
	//	Mat image_lateralInvert(frame.size(), frame.type());
	//	remap(frame, image_lateralInvert, map_x, map_y, CV_INTER_LINEAR);
	//	frame = image_lateralInvert;
	//}

#ifdef DURATION_CHECK
	//time = (cv::getTickCount() - curCount) / cv::getTickFrequency();
	//printf("invert Duration: %f\n", time);
#endif

	cv::Mat canny;
	Mat image_gray_blur;
	Mat channel[3];
	split(frame, channel);
	grayOut = channel[2].clone();
	//cvtColor(frame, grayOut, CV_BGR2GRAY);  // Convert to grayscale image

	GaussianBlur(grayOut, image_gray_blur, cv::Size(5, 5), 2, 2);  // Reduce the noise
	//blur(grayOut, image_gray_blur, cv::Size(3, 3));  // Reduce the noise

#ifdef DURATION_CHECK
	//time = (cv::getTickCount() - curCount) / cv::getTickFrequency();
	//printf("Blur Duration: %f\n", time);
#endif

	// Create mask for edges
	Canny(image_gray_blur, canny, lowThreshold, (double)(lowThreshold)*CANNY_MAX_THRHD_RATIO); // Apply Canny detector
	cv::Mat dilate_element = cv::getStructuringElement(MORPH_ELLIPSE, Size(dilateSize, dilateSize), Point(-1, -1));
	dilate(canny, canny, dilate_element);

#ifdef SHOW_IMAGE_AND_RESULT
	// Use Canny's output as a mask and display the canny's result
	//cv::Mat dst(frame.size(), frame.type(), Scalar::all(0));
	//frame.copyTo(dst, canny);
	//cv::imshow(WINDOW_NAME_EDGE_MASK, dst);
	//cv::imwrite("_edge_mask.bmp", dst);
	cv::imwrite("_edge_mask.bmp", canny);
	waitKey(100);
#endif

	// Finds contourts from the frame
	cv::findContours(canny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	return ;
}

cv::Mat* SignRecognizer::invertLaterally(cv::Mat *sign)
{
	cv::Mat mapX, mapY;
	cv::Mat signInverted(sign->size(), sign->type());
	mapX.create(sign->size(), CV_32FC1);
	mapY.create(sign->size(), CV_32FC1);

	for (int i = 0; i < sign->size().height; i++)
	{
		for (int j = 0; j < sign->size().width; j++)
		{
			mapX.at<float>(i, j) = (float)(sign->size().width - j);
			mapY.at<float>(i, j) = (float)(i);
		}
	}

	remap(*sign, signInverted, mapX, mapY, CV_INTER_LINEAR);
	*sign = signInverted.clone();
	return sign;
}

void SignRecognizer::updateMap(void)
{
	map_x.create(frameSize, CV_32FC1);
	map_y.create(frameSize, CV_32FC1);
	for (int i = 0; i < frameSize.height; i++)
	{
		for (int j = 0; j < frameSize.width; j++)
		{
			map_x.at<float>(i, j) = (float)(frameSize.width - j);
			map_y.at<float>(i, j) = (float)(i);
		}
	}

	return;
}

void SignRecognizer::updateMap(cv::Size size)
{
	map_x.create(size, CV_32FC1);
	map_y.create(size, CV_32FC1);
	for (int i = 0; i < size.height; i++)
	{
		for (int j = 0; j < size.width; j++)
		{
			map_x.at<float>(i, j) = (float)(size.width - j);
			map_y.at<float>(i, j) = (float)(i);
		}
	}

	return;
}

void SignRecognizer::testExample(void)
{
	//Mat src = imread("myImages/20160105_194622.jpg", CV_LOAD_IMAGE_COLOR);
	//Mat src = imread("myImages/20160105_223137.jpg", CV_LOAD_IMAGE_COLOR);
	//Mat src = imread("myImages/20160105_223215.jpg", CV_LOAD_IMAGE_COLOR);
	//Mat src = imread("myImages/20160105_223241.jpg", CV_LOAD_IMAGE_COLOR);
	//setFrameSize(src.size().width, src.size().height);
	//isLaterallyInverted = false;

	//Mat src = imread("myImages/color34.bmp", CV_LOAD_IMAGE_COLOR);
	//Mat src = imread("myImages/color38.bmp", CV_LOAD_IMAGE_COLOR);
	std::stringstream oss;





	oss.str("");
	oss << "bye2/132_colour.png";
	Mat src = imread(oss.str(), CV_LOAD_IMAGE_COLOR);
	if (!src.data)
		return;

	oss.str("");
	oss << "bye2/result/_yeah_";
	this->runRecognizer(src, oss.str());





	//for (int i = 0; i < 1000; i++)
	//{
	//	oss.str("");
	//	oss << "bye2/" << i << "_colour.png";
	//	Mat src = imread(oss.str(), CV_LOAD_IMAGE_COLOR);
	//	if (!src.data)
	//	{
	//		continue;

	//	}
	//	oss.str("");
	//	oss << "bye2/result/" << i;
	//	this->runRecognizer(src, oss.str());
	//}



	//for (currSet = 9; currSet <= 9; currSet++)
	//{
	//	oss.str("");
	//	oss << "_Test_Sign_" << currSet << "/result/result.csv";
	//	fout.open(oss.str(), std::fstream::out);
	//	if (!fout.is_open())
	//	{
	//		std::cout << "Cannot open result.csv" << std::endl;
	//		system("pause");
	//		return;
	//	}
	//	fout << "Frame No,detected text,floor,isCorr" << std::endl;

	//	for (int i = 0; i < 300; i++)
	//	{
	//		oss.str("");
	//		oss << "_Test_Sign_" << currSet << "/" << i << "_image.bmp";
	//		Mat src = imread(oss.str(), CV_LOAD_IMAGE_COLOR);
	//		if (!src.data)
	//		{
	//			//std::cout << oss.str() << " is not loaded" << std::endl;
	//			continue;

	//		}
	//		oss.str("");
	//		oss << "_Test_Sign_" << currSet << "/result/" << i;
	//		this->runRecognizer(src, oss.str());
	//	}

	//	fout.close();
	//}
	std::cout << "Finished" << std::endl;
}
