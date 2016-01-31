/* ----------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
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

using namespace cv;

const std::string SignRecognizer::STRING_GROUND_FLOOR = "Ground floor";
const std::string SignRecognizer::STRING_FIRST_FLOOR = "first floor";
const std::string SignRecognizer::STRING_SECOND_FLOOR = "second floor";
const std::string SignRecognizer::STRING_THIRD_FLOOR = "third floor";
const std::string SignRecognizer::STRING_FOURTH_FLOOR = "fourth floor";
const std::string SignRecognizer::STRING_FIFTH_FLOOR = "fifth floor";
const std::string SignRecognizer::STRING_SIXTH_FLOOR = "sixth floor";
const std::string SignRecognizer::STRING_SEVEN_FLOOR = "seven floor";
const std::string SignRecognizer::STRING_LG1 = "LG1";
const std::string SignRecognizer::STRING_LG2 = "LG2";
const std::string SignRecognizer::STRING_LG3 = "LG3";
const std::string SignRecognizer::STRING_LG4 = "LG4";
const std::string SignRecognizer::STRING_LG5 = "LG5";
const std::string SignRecognizer::STRING_LG6 = "LG6";
const std::string SignRecognizer::STRING_LG7 = "LG7";

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
	dilateSize = 1;
	
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

void SignRecognizer::runRecognizer(cv::Mat &frame)
{
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
		{
#ifdef SHOW_DEBUG_MESSAGES
			std::cout << "small / big contour detected..";
			std::cout << std::endl;
#endif
			continue;
		}

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
			OutputDebugStringA("Contour[");
			OutputDebugStringA(std::to_string(k).c_str());
			OutputDebugStringA("] VS last SavedContour, distance = ");
			OutputDebugStringA(std::to_string(distance).c_str());
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

		RotatedRect rrect = fitEllipse(contours_all[k]);
		Mat foundEllipse(frameGray.size(), frameGray.type(), Scalar::all(0));

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
		OutputDebugStringA("SavedContour[");
		OutputDebugStringA(std::to_string(k).c_str());
		OutputDebugStringA("] & estimated ellipse, distance = ");
		OutputDebugStringA(std::to_string(distance).c_str());
#endif

		// Determine whether there is any contour of ellipse
		if (distance > SAME_ELLIPSE_THRESHOLD) { // if there is an ellipse
#ifdef SHOW_DEBUG_MESSAGES
			std::cout << " => Ellipse Found\n";
			OutputDebugStringA(" => Ellipse Found\n");
#endif

#ifdef SHOW_MARKERS
			ellipse(frame, rrect, Scalar(0, MAX_VALUE_8BITS/2, MAX_VALUE_8BITS), 2); // Draw the ellipse on the image
#endif

			// Binarize the sign
			sign = new Mat(frameGray(rect) & ellipseMask);
			avg_tmp = cv::mean(*sign, ellipseMask)[0];
			avg_tmp *= AVG_GRAYSCALE_SCALE;

#ifdef SHOW_DEBUG_MESSAGES
			std::cout << "Grayscale average: " << avg_tmp << std::endl;
			OutputDebugStringA("Grayscale average: ");
			OutputDebugStringA(std::to_string(avg_tmp).c_str());
			OutputDebugStringA("\n");
#endif

			cv::Mat ellipseMask_inv(ellipseMask.size(), ellipseMask.type(), Scalar::all(0));
			cv::bitwise_not(ellipseMask, ellipseMask_inv);
			cv::threshold(*sign, *sign, avg_tmp, ABSOLUTE_WHITE, THRESH_BINARY);
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
			medianBlur(*sign, *sign, MEDIAN_BLUR_KSIZE);

			// apply OCR to obtain the characters
			tess.SetImage((uchar*)(sign->data), sign->cols, sign->rows, 1, sign->step1());
			string text = tess.GetUTF8Text();
			text.erase(remove(text.begin(), text.end(), '\n'), text.end());
			text.erase(remove(text.begin(), text.end(), ' '), text.end());
#ifdef SHOW_DEBUG_MESSAGES
			std::cout << "SavedContour[" << k << "] text: " << text << "_Floor";
			OutputDebugStringA("SavedContour[");
			OutputDebugStringA(std::to_string(k).c_str());
			OutputDebugStringA("] text: ");
			OutputDebugStringA(text.c_str());
			OutputDebugStringA("_Floor");
#endif

			std::wstring out;
			if (!getResultString(text, out)) {
				std::cerr << std::endl << "Error on getResultString()" << std::endl;
				continue;
			}

			cv::putText(frame, text, contours_all[k][1], cv::FONT_HERSHEY_SIMPLEX, 1,
				cv::Scalar(0, 128, 255));

			TextToSpeech::pushBack(out);
			//tts.pushBack(out);
#ifdef SPEAK_OUT_RESULT
			tts.speak();
#endif

#ifdef SHOW_DEBUG_MESSAGES
			std::cout << std::endl;
			OutputDebugStringA("\n");
#endif

			savedSigns_index.push_back(k);

#ifdef SHOW_IMAGE_AND_RESULT
			// Save and show the sign
			savedSigns.push_back(*sign);

			sout.str("");
			sout << "_sign_" << k << ".jpg";
			namedWindow(sout.str(), CV_WINDOW_AUTOSIZE);
			cv::imshow(sout.str(), *sign);

#ifdef SAVE_IMAGE_AND_RESULT
			cv::imwrite(sout.str(), sign);
#endif
#endif

#ifdef SHOW_IMAGE_AND_RESULT
			sout.str("");
			sout << "Contours " << k;
			namedWindow(sout.str(), CV_WINDOW_AUTOSIZE);
			cv::imshow(sout.str(), contourImg);
#endif

			sign = NULL;
		}


	}
#ifdef SHOW_DEBUG_MESSAGES
	std::cout << std::endl << "No of Contours saved = " << savedSigns_index.size() << std::endl;
	OutputDebugStringA("] VS last SavedContour, distance = ");
	OutputDebugStringA(std::to_string(savedSigns_index.size()).c_str());
	OutputDebugStringA("\n");
#endif

#ifdef SHOW_IMAGE_AND_RESULT
#ifdef SAVE_IMAGE_AND_RESULT
	// Show and save the result
	sout.str("");
	sout << "_result.jpg";
	cv::imwrite(sout.str(), frame);
#endif

	namedWindow("Detection Result", CV_WINDOW_NORMAL);
	cv::imshow("Detection Result", frame);
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
	if (strcmp(in.c_str(), "G") == 0)
	{
		out = std::wstring(STRING_GROUND_FLOOR.begin(), STRING_GROUND_FLOOR.end());
	}
	else if (strcmp(in.c_str(), "1") == 0)
	{
		out = std::wstring(STRING_FIRST_FLOOR.begin(), STRING_FIRST_FLOOR.end());
	}
	else if (strcmp(in.c_str(), "2") == 0)
	{
		out = std::wstring(STRING_SECOND_FLOOR.begin(), STRING_SECOND_FLOOR.end());
	}
	else if (strcmp(in.c_str(), "3") == 0)
	{
		out = std::wstring(STRING_THIRD_FLOOR.begin(), STRING_THIRD_FLOOR.end());
	}
	else if (strcmp(in.c_str(), "4") == 0)
	{
		out = std::wstring(STRING_FOURTH_FLOOR.begin(), STRING_FOURTH_FLOOR.end());
	}
	else if (strcmp(in.c_str(), "5") == 0)
	{
		out = std::wstring(STRING_FIFTH_FLOOR.begin(), STRING_FIFTH_FLOOR.end());
	}
	else if (strcmp(in.c_str(), "6") == 0)
	{
		out = std::wstring(STRING_SIXTH_FLOOR.begin(), STRING_SIXTH_FLOOR.end());
	}
	else if (strcmp(in.c_str(), "7") == 0)
	{
		out = std::wstring(STRING_SEVEN_FLOOR.begin(), STRING_SEVEN_FLOOR.end());
	}
	else
	{
		out = std::wstring(in.begin(), in.end());
	}

	return true;
}

void SignRecognizer::getContoursOfFrame(cv::Mat &frame, cv::Mat &grayOut, std::vector<std::vector<Point>> &contours, std::vector<Vec4i> &hierarchy)
{
	if (isLaterallyInverted)
	{
		Mat image_lateralInvert(frame.size(), frame.type());
		remap(frame, image_lateralInvert, map_x, map_y, CV_INTER_LINEAR);
		frame = image_lateralInvert;
	}

	Mat canny;
	Mat image_gray_blur;

	cvtColor(frame, grayOut, CV_BGR2GRAY);  // Convert it to grayscale image
	GaussianBlur(grayOut, image_gray_blur, cv::Size(9, 9), 2, 2);  // Reduce the noise

	Canny(image_gray_blur, canny, lowThreshold, (double)(lowThreshold)*CANNY_MAX_THRHD_RATIO); // Apply Canny detector

	// Create mask for edges by dilation
	cv::Mat dilate_element = cv::getStructuringElement(MORPH_ELLIPSE, Size(dilateSize, dilateSize), Point(-1, -1));
	dilate(canny, canny, dilate_element);

#ifdef SHOW_IMAGE_AND_RESULT
	// Use Canny's output as a mask and display the canny's result
	cv::Mat dst(frame.size(), frame.type(), Scalar::all(0));
	frame.copyTo(dst, canny);
	cv::imshow(WINDOW_NAME_EDGE_MASK, dst);
#endif

	// Finds contourts from the frame
	cv::findContours(canny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	return ;
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
	Mat src = imread("myImages/color38.bmp", CV_LOAD_IMAGE_COLOR);
	
	if (!src.data)
		std::cout << "no image is loaded" << std::endl;
	else
		this->runRecognizer(src);
}
