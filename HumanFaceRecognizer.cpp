// HumanFaceRecognizer.cpp : Defines the entry point for the console application.
//

/* Includes */
#include "HumanFaceRecognizer.h"


/* Global Variables */
//const std::string PERSON_NAME[NUM_OF_PERSON + 1] =
//{
//	"Guest",
//	"Joel",
//	"Ka Ho",
//	"Yumi"
//};

int image_num = 0;


/* Functions */
HumanFaceRecognizer::HumanFaceRecognizer()
{
	model = cv::createLBPHFaceRecognizer();
	model->load(DB_FACE_FILE_PATH);

	total_percent = 0.0;
	total_percent_var = 0.0;
	min_percent = 0.35;
	max_percent = 0.65;
	num_of_face_detected = 0;

	num_of_person_in_db = 0;

	std::ifstream fin;
	fin.open(DB_NAME_FILE_PATH);
	if (!fin.is_open())
	{
		std::cerr << "Cannot open " << DB_NAME_FILE_PATH << "for people names" << std::endl;
		exit(0);
	}

	std::string buf;
	while (!fin.eof())
	{
		std::getline(fin, buf, ',');
		std::getline(fin, buf);
		std::cout << buf << std::endl;
		
		if (!buf.empty())
			PERSON_NAME.push_back(buf);
	}
	num_of_person_in_db = PERSON_NAME.size();
	std::cout << "num_of_person_in_db: " << num_of_person_in_db << std::endl;
	//PERSON_NAME.push_back("Guest");
	//PERSON_NAME.push_back("Joel");
	//PERSON_NAME.push_back("Ka Ho");
	//PERSON_NAME.push_back("Yumi");
	fin.close();


	isUpdated = false;
	isAddNewFace = false;
}

HumanFaceRecognizer::~HumanFaceRecognizer()
{

}

int HumanFaceRecognizer::runFaceRecognizer(cv::Mat *frame)
{
#ifdef RESIZE_TO_SMALLER
	cv::Mat original = detector.resizeToSmaller(frame);
#else
	cv::Mat original = (*frame).clone();
#endif

#ifdef COMPARE_FACE_COLOUR
	cv::Mat outputMask;
#endif

	double face_pixel_num = 0;
	double similar_pixel_counter = 0;
	int i, j, k, p, q, r;
	int face_num = 0; // variable used when saving faces or masks
	std::vector<cv::Rect> newFacePos;
	std::ostringstream oss;


	int face_counter1 = 0;
	int face_counter2 = 0;
	int predictedLabel = -1;
	double confidence = 0.0;
	double confidence_threshold = 100;
	bool isExistedFace = false;

	// Apply the classifier to the frame
	detector.getFaces(*frame, newFacePos);
	cv::vector<cv::Rect>::iterator it = newFacePos.begin();


	// Reject one of the detected faces if the positions of two faces are too closed
	// Reject the one with fewer number of detected faces over the time at its position
	for (p = 1; p < (int)(facesInfo.size()); ++p)
	{
		for (q = 0; (p != q) && (q < p) && p < (int)(facesInfo.size()); ++q)
		{
			if (abs(facesInfo[p].centerPos.x - facesInfo[q].centerPos.x) < FACE_POS_OFFSET &&
				abs(facesInfo[p].centerPos.y - facesInfo[q].centerPos.y) < FACE_POS_OFFSET)
			{
				face_counter1 = 0;
				face_counter2 = 0;
				for (r = 0; r <= NUM_OF_PERSON; ++r)
				{
					face_counter1 += facesInfo[p].counter[r];
					face_counter2 += facesInfo[q].counter[r];
				}
#ifdef SHOW_DEBUG_MESSAGES
				std::cout << std::endl << face_counter1 << "\t" << face_counter2 << "\t";
				std::cout << p << "\t" << q << std::endl;
#endif

				if (face_counter1 > face_counter2)
				{
					if (facesInfo[q].counter[facesInfo[p].label] > 0)
					{
						facesInfo[p].counter[facesInfo[p].label] += facesInfo[q].counter[facesInfo[p].label];
						facesInfo[p].centerPos.x = facesInfo[q].centerPos.x;
						facesInfo[p].centerPos.y = facesInfo[q].centerPos.y;
					}
					facesInfo.erase(facesInfo.begin() + q);
				}
				else if (face_counter1 < face_counter2)
				{
					if (facesInfo[p].counter[facesInfo[q].label] > 0)
					{
						facesInfo[q].counter[facesInfo[q].label] += facesInfo[p].counter[facesInfo[q].label];
						facesInfo[q].centerPos.x = facesInfo[p].centerPos.x;
						facesInfo[q].centerPos.y = facesInfo[p].centerPos.y;
					}
					facesInfo.erase(facesInfo.begin() + p);
				}
			}
		}
	}

	// If a detected face at certain position is not detected for a period of time, it is discarded
	for (p = 0; p < (int)facesInfo.size(); ++p)
	{
		if (facesInfo[p].undetected_counter > UNDETECTED_THREHOLD)
		{
#ifdef SHOW_DEBUG_MESSAGES
			std::cout << "erase: " << p << std::endl;
#endif
#ifdef SHOW_MARKERS
			oss.str("");
			oss << "ERASE";
			putText(*frame, oss.str(), facesInfo[p].centerPos, cv::FONT_HERSHEY_SIMPLEX, 0.6,
				cv::Scalar(0, 128, 255), 2);
#endif

			facesInfo.erase(facesInfo.begin() + p--);
			continue;
		}

		for (it = newFacePos.begin(); it != newFacePos.end(); ++it)
		{
			cv::Point center(it->x + it->width / 2, it->y + it->height / 2);

			if ((abs(facesInfo[p].centerPos.x - center.x) < FACE_POS_OFFSET) &&
				(abs(facesInfo[p].centerPos.y - center.y) < FACE_POS_OFFSET))
				break;
		}

		if (it == newFacePos.end())
		{
			++(facesInfo[p].undetected_counter);
#ifdef SHOW_DEBUG_MESSAGES
			std::cout << "undetected: " << facesInfo[p].undetected_counter << std::endl;
#endif
#ifdef SHOW_MARKERS
			oss.str("");
			oss << "++";
			putText(*frame, oss.str(), facesInfo[p].centerPos, cv::FONT_HERSHEY_SIMPLEX, 0.6,
				cv::Scalar(0, 128, 255), 2);
#endif
		}

	}

	// evaluate a list of possible faces
	for (i = 0, it = newFacePos.begin(); it != newFacePos.end(); ++it, ++i)
	{
		++face_num;

#ifdef RESIZE_TO_SMALLER
		cv::Mat face = original(cv::Rect((*it).x * RESIZE_SCALE, (*it).y * RESIZE_SCALE, 
			(*it).width * RESIZE_SCALE, (*it).height * RESIZE_SCALE)).clone();
#else
		cv::Mat face = original(*it).clone();
#endif
		cv::Mat face_grey;
		cv::Point center(it->x + it->width*0.5, it->y + it->height*0.5);
		cv::Point top(it->x, it->y);

#ifdef COMPARE_FACE_COLOUR
		cv::vector<cv::Mat> channels;
		cv::Mat face_eq;
		cvtColor(face, face_eq, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format
		split(face_eq, channels); //split the image into channels
		equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)
		merge(channels, face_eq); //merge 3 channels including the modified 1st channel into one image
		cvtColor(face_eq, face_eq, CV_YCrCb2BGR); //change the color image from YCrCb to BGR format (to display image properly)

		detector.compareFaceColour(face_eq, outputMask);
		//detector.compareFaceColour(face, outputMask);

#ifndef FACE_MASK_COLOUR
		face_pixel_num = outputMask.rows * outputMask.cols;
#else
		face_pixel_num = outputMask.rows * outputMask.cols * NUM_OF_CHANNELS_COLOUR;
#endif
		for (j = 0; j < outputMask.rows; ++j)
		{
			for (k = 0; k < outputMask.cols; ++k)
			{
#ifndef FACE_MASK_COLOUR
				if (*(outputMask.data + (j*outputMask.cols + k)) == 255)
					similar_pixel_counter += 1;
#else
				for (int m = 0; m < NUM_OF_CHANNELS_COLOUR; ++m)
				{
					if (*(outputMask.data + j*outputMask.step + k + m) == 255)
						similar_pixel_counter += 1;
				}
#endif
			}
		}
		similar_pixel_counter /= face_pixel_num;
#endif

#ifdef COMPARE_FACE_COLOUR
		cv::cvtColor(face_eq, face_grey, CV_BGR2GRAY);
		if ((similar_pixel_counter > min_percent) && (similar_pixel_counter < max_percent))  // if the percentage of similar pixeel is within certain range, it is a face
#else
		cv::cvtColor(face, face_grey, CV_BGR2GRAY);
#endif
		{
			cv::imwrite("face_grey.bmp", face_grey);
			model->predict(face_grey, predictedLabel, confidence);
			if (confidence > confidence_threshold)
				predictedLabel = Guest;

			isExistedFace = false;
			for (p = 0; p < facesInfo.size(); ++p)
			{
				if (isExistedFace)
					break;

				if ((abs(facesInfo[p].centerPos.x - center.x) < FACE_POS_OFFSET) &&
					(abs(facesInfo[p].centerPos.y - center.y) < FACE_POS_OFFSET))
				{
					memcpy(&(facesInfo[p].centerPos), &center, sizeof(cv::Point));
					++(facesInfo[p].counter[predictedLabel]);

					if (!(facesInfo[p].isRecognized))
					{
						std::string str;
						oss.str("");
						switch (predictedLabel)
						{
						case -1:
#ifdef SHOW_MARKERS
							oss << "unrecognised";
#endif
							break;

						case Guest:
							if (facesInfo[p].counter[Guest] >= FACE_DET_THREHOLD * 2) {
#ifdef SHOW_MARKERS
								oss << PERSON_NAME[Guest] << " " << confidence;
#endif
								if (facesInfo[p].counter[Guest] == 10) {
									str = std::string(HELLO_MESSAGE) + std::string(PERSON_NAME[Guest]);
									TextToSpeech::pushBack(str);
								}
							}
#ifdef SHOW_MARKERS
							else
								oss << DETECTING << confidence;
#endif
							break;

						case Joel:
						case KaHo:
						case Yumi:
						default:
							if (facesInfo[p].counter[predictedLabel] >= FACE_DET_THREHOLD) {
#ifdef SHOW_MARKERS
								//oss << PERSON_NAME[facesInfo[p].label] << " " << confidence;
								oss << PERSON_NAME[predictedLabel] << " detected";
#endif
								facesInfo[p].isRecognized = true;
								facesInfo[p].label = (DETECTED_PERSON)predictedLabel;
#ifdef SHOW_DEBUG_MESSAGES
								std::cout << "detected: " << predictedLabel << '\n';
#endif
								/* Text to Speech */
								str = std::string(HELLO_MESSAGE) + std::string(PERSON_NAME[predictedLabel]);
								TextToSpeech::pushBack(str);
							}
#ifdef SHOW_MARKERS
							else
							{
								//oss << DETECTING << ", maybe " << PERSON_NAME[facesInfo[p].label];
								oss << "maybe " << PERSON_NAME[predictedLabel] << "-" << confidence;
							}
#endif
							break;
						}

					}
#ifdef SHOW_MARKERS
					else
					{
						oss.str("");
						oss << "D:" << PERSON_NAME[facesInfo[p].label] << ",R:" << PERSON_NAME[predictedLabel] << "-" << confidence;

						//switch (facesInfo[p].label)
						//{
						//case Guest:
						//case Joel:
						//case KaHo:
						//case Yumi:
						//	//oss << PERSON_NAME[facesInfo[p].label] << "-" << confidence;
						//	oss << "D:" << PERSON_NAME[facesInfo[p].label] << ",R:" << PERSON_NAME[predictedLabel] << "-" << confidence;
						//	//oss << PERSON_NAME[facesInfo[p].label];
						//	break;

						//default:
						//	oss << "Special!! " << confidence;
						//	//oss << NAME_GUEST;
						//	break;
						//}
					}
#endif
#ifdef SHOW_MARKERS
					putText(*frame, oss.str(), top, cv::FONT_HERSHEY_SIMPLEX, 0.5,
						cv::Scalar(0, 0, 255), 1);
#endif
					isExistedFace = true;
				}
			}

			if (facesInfo.size() == 0 || !isExistedFace)
			{
				DetectionInfo para;
				memset(&para, 0, sizeof(DetectionInfo));
				para.isRecognized = false;
				memcpy(&(para.centerPos), &center, sizeof(cv::Point));
				memcpy(&(para.size), &(it->size()), sizeof(cv::Size));
				para.counter.resize(num_of_person_in_db, 0);
				para.counter[predictedLabel] = 1;
				facesInfo.push_back(para);

#ifdef SHOW_MARKERS
				oss.str("");
				oss << "maybe " << PERSON_NAME[predictedLabel] << "-" << confidence;
				putText(*frame, oss.str(), top, cv::FONT_HERSHEY_SIMPLEX, 0.5,
					cv::Scalar(255, 0, 255, 1));
#endif
			}
#ifdef SHOW_DEBUG_MESSAGES
			std::cout << "facesInfo size: " << facesInfo.size() << std::endl;
#endif

#ifdef SHOW_MARKERS
			ellipse(*frame, center, cv::Size(it->width*0.5, it->height*0.5), 0, 0, 360, cv::Scalar(0, 0, 255), 4, 8, 0);
#endif

#ifdef SAVE_IMAGES
#ifdef SAVE_FACES
			oss.str("");
			oss << BASE_DIR << CORRECT_DIR << image_num << "_" << face_num << FACE_NAME_POSTFIX << IMAGE_EXTENSION;
			imwrite(oss.str(), face);
#endif
#ifdef COMPARE_FACE_COLOUR
#ifdef SAVE_MASKS
			oss.str("");
			oss << BASE_DIR << CORRECT_DIR << image_num << "_" << face_num << MASK_NAME_POSTFIX << IMAGE_EXTENSION;
			imwrite(oss.str(), outputMask);
#endif
#endif
#endif
		}
#ifdef COMPARE_FACE_COLOUR
		else // it is not a face
		{
#ifdef SHOW_MARKERS
			ellipse(*frame, center, cv::Size(it->width*0.5, it->height*0.5), 0, 0, 360, cv::Scalar(255, 0, 0), 4, 8, 0);
#endif

#ifdef SAVE_IMAGES
#ifdef SAVE_FACES
			oss.str("");
			oss << BASE_DIR << WRONG_DIR << image_num << "_" << face_num << FACE_NAME_POSTFIX << IMAGE_EXTENSION;
			imwrite(oss.str(), face);
#endif
#ifdef SAVE_MASKS
			oss.str("");
			oss << BASE_DIR << WRONG_DIR << image_num << "_" << face_num << MASK_NAME_POSTFIX << IMAGE_EXTENSION;
			imwrite(oss.str(), outputMask);
#endif
#endif
		}
#endif


#ifdef DISPLAY_FACES_AND_MASKS
		oss.str("");
		oss << "face[" << i << "]";
		cv::namedWindow(oss.str());                        // Create a window for display.
		cv::imshow(oss.str(), face);                       // Show our image inside it.

#ifdef COMPARE_FACE_COLOUR
		oss.str("");
		oss << "outputMask[" << i << "]";
		cv::namedWindow(oss.str());                        // Create a window for display.
		cv::imshow(oss.str(), outputMask);                 // Show our image inside it.
#endif
#endif

		num_of_face_detected++;
		total_percent += similar_pixel_counter;
		//total_percent_var += pow(similar_pixel_counter - total_percent, 2);

		similar_pixel_counter = 0;
	}

#ifdef SHOW_MARKERS
	oss.str("");
	oss << facesInfo.size();
	putText(*frame, oss.str(), cv::Size(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
#endif

#ifdef SAVE_IMAGES
	//oss.str("");
	//oss << IMAGE_DIR << image_num << IMAGE_NAME_POSTFIX << IMAGE_EXTENSION;
	//imwrite(oss.str(), original);

	oss.str("");
	oss << IMAGE_DIR << "frame_" << image_num << IMAGE_NAME_POSTFIX << IMAGE_EXTENSION;
	imwrite(oss.str(), *frame);
#endif
	++image_num;


#ifdef DISPLAY_IMAGES
	cv::namedWindow("Video_Stream", CV_WINDOW_AUTOSIZE);     // Create a window for display.
	cv::imshow("Video_Stream", *frame);
#endif

	return 0;
}


void HumanFaceRecognizer::addNewFace(cv::Mat &frame, std::string name)
{
#ifdef RESIZE_TO_SMALLER
	cv::Mat original = detector.resizeToSmaller(&frame);
#else
	cv::Mat original = (*frame).clone();
#endif

	facesInfo.clear();

	std::vector<cv::Mat> faceList;
	std::vector<int> labelList;
	
	std::vector<cv::Rect> facePos;
	detector.getFaces(frame, facePos); // Apply the classifier to the frame
	if (facePos.size() == 0)
	{
		std::cerr << "Cannot see any face within the camera!\n";
		return;
	}

	int realFaceIndex = 0;
	for (int i = 0; i < facePos.size(); i++)
	{
		if (facePos[realFaceIndex].size().width < facePos[i].size().width)
			realFaceIndex = i;
	}

#ifdef SHOW_MARKERS
	ellipse(frame, cv::Point(facePos[realFaceIndex].x + facePos[realFaceIndex].size().width / 2,
		facePos[realFaceIndex].y + facePos[realFaceIndex].size().height / 2),
		cv::Size(facePos[realFaceIndex].size().width*0.5, facePos[realFaceIndex].size().height*0.5),
		0, 0, 360, cv::Scalar(0, 0, 255), 4, 8, 0);
#endif


#ifdef RESIZE_TO_SMALLER
	cv::Mat face_r = original(cv::Rect(facePos[realFaceIndex].x * RESIZE_SCALE,
		facePos[realFaceIndex].y * RESIZE_SCALE,
		facePos[realFaceIndex].width * RESIZE_SCALE,
		facePos[realFaceIndex].height * RESIZE_SCALE)).clone();
#else
	cv::Mat face_r = original(facePos[realFaceIndex]).clone();
#endif

	//cv::Mat face_r = frame(facePos[realFaceIndex]).clone();
	Mat mask(cv::Size(face_r.size()), face_r.type(), Scalar::all(0));

	circle(mask, Point(face_r.size().width / 2, face_r.size().height / 2),
		face_r.size().width / 2, Scalar::all(255), -1);

	Mat face = face_r & mask; // combine roi & mask
	imshow("NEW FACE", face);


	// Equalization
	vector<Mat> channels;
	Mat face_eq;
	cvtColor(face, face_eq, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format
	split(face_eq, channels); //split the image into channels
	equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)
	merge(channels, face_eq); //merge 3 channels including the modified 1st channel into one image
	cvtColor(face_eq, face_eq, CV_YCrCb2BGR); //change the color image from YCrCb to BGR format (to display image properly)
	cvtColor(face_eq, face_eq, CV_RGB2GRAY);

	faceList.push_back(face_eq);

	// update database
	int label = -1;
	int i;
	std::string tmp1 = name;
	std::transform(tmp1.begin(), tmp1.end(), tmp1.begin(), ::tolower);
	for (i = 0; i < PERSON_NAME.size(); i++)
	{
		std::string tmp2 = PERSON_NAME[i];
		std::transform(tmp2.begin(), tmp2.end(), tmp2.begin(), ::tolower);

		if (strcmp(tmp1.c_str(), tmp2.c_str()) == 0)
		{
			label = i;
			break;
		}
	}

	if (i == PERSON_NAME.size())
	{
		label = i;
		PERSON_NAME.push_back(name);
		num_of_person_in_db++;
	}


	if (label != -1)
	{
		labelList.push_back(label);
		model->update(faceList, labelList);
	}

	return;
}

void HumanFaceRecognizer::testExample(void)
{
	// Initialize Camera
	cv::VideoCapture camera1;
	camera1.open(0);

	cv::Mat frame;
	while (true)
	{
		camera1 >> frame;
		runFaceRecognizer(&frame);

		if (cv::waitKey(1) == (int)'0')
			break;
	}
	total_percent /= (double)(num_of_face_detected);
	std::cout << "Avg percent: " << total_percent << std::endl;

	cv::destroyAllWindows();
}
