// HumanFaceRecognizer.cpp : Defines the entry point for the console application.
//

/* Includes */
#include "HumanFaceRecognizer.h"


/* Global Variables */
const std::string PERSON_NAME[NUM_OF_PERSON + 1] =
{
	"Guest",
	"Joel",
	"Ka Ho",
	"Yumi"
};

#ifdef SAVE_IMAGES
	int image_num = 0;
#endif


/* Functions */
HumanFaceRecognizer::HumanFaceRecognizer()
{
	model = cv::createLBPHFaceRecognizer();
	model->load(DB_FILE_PATH);

	total_percent = 0.0;
	total_percent_var = 0.0;
	min_percent = 0.4;
	max_percent = 0.85;
	num_of_face_detected = 0;
}

HumanFaceRecognizer::~HumanFaceRecognizer()
{

}

int HumanFaceRecognizer::runFaceRecognizer(cv::Mat *frame)
{
#ifdef COMPARE_FACE_COLOUR
	cv::Mat outputMask;
#endif

	double face_pixel_num = 0;
	double similar_pixel_counter = 0;
	int i, j, k, p, q, r;
	int face_num = 0; // variable used when saving faces or masks
	std::vector<cv::Rect> newFacePos;
	//Mat face;
	cv::Mat face_grey;
	cv::Mat original;
	std::ostringstream oss;


	int face_counter1 = 0;
	int face_counter2 = 0;
	int predictedLabel = -1;
	double confidence = 0.0;
	double confidence_threshold = 120;
	bool isExistedFace = false;

	original = (*frame).clone();

	// Apply the classifier to the frame
	detector.getFaces(*frame, newFacePos);
	cv::vector<cv::Rect>::iterator it = newFacePos.begin();

	// Reject one of the detected faces if the positions of two faces are too closed
	// Reject the one with fewer number of detected faces over the time at its position
	for (p = 1; p < (int)(facesInfo.size()); ++p)
	{
		for (q = 0; (p != q) && (q < p) && p < (int)(facesInfo.size()); ++q)
		{
			if (abs(facesInfo[p].facePos.x - facesInfo[q].facePos.x) < FACE_POS_OFFSET &&
				abs(facesInfo[p].facePos.y - facesInfo[q].facePos.y) < FACE_POS_OFFSET)
			{
				face_counter1 = 0;
				face_counter2 = 0;
				for (r = 0; r <= NUM_OF_PERSON; ++r)
				{
					face_counter1 += facesInfo[p].face_counter[r];
					face_counter2 += facesInfo[q].face_counter[r];
				}
#ifdef SHOW_DEBUG_MESSAGES
				std::cout << std::endl << face_counter1 << "\t" << face_counter2 << "\t";
				std::cout << p << "\t" << q << std::endl;
#endif

				if (face_counter1 > face_counter2)
				{
					if (facesInfo[q].face_counter[facesInfo[p].label] > 0)
					{
						facesInfo[p].face_counter[facesInfo[p].label] += facesInfo[q].face_counter[facesInfo[p].label];
						facesInfo[p].facePos.x = facesInfo[q].facePos.x;
						facesInfo[p].facePos.y = facesInfo[q].facePos.y;
					}
					facesInfo.erase(facesInfo.begin() + q);
				}
				else if (face_counter1 < face_counter2)
				{
					if (facesInfo[p].face_counter[facesInfo[q].label] > 0)
					{
						facesInfo[q].face_counter[facesInfo[q].label] += facesInfo[p].face_counter[facesInfo[q].label];
						facesInfo[q].facePos.x = facesInfo[p].facePos.x;
						facesInfo[q].facePos.y = facesInfo[p].facePos.y;
					}
					facesInfo.erase(facesInfo.begin() + p);
				}
			}
		}
	}

	// If a detected face at certain position is not detected for a period of time, it is discarded
	for (p = 0; p < (int)facesInfo.size(); ++p)
	{
		for (it = newFacePos.begin(); it != newFacePos.end(); ++it)
		{
			if ((facesInfo[p].facePos.x >(it->x - FACE_POS_OFFSET)) && (facesInfo[p].facePos.x < (it->x + FACE_POS_OFFSET)) &&
				(facesInfo[p].facePos.y >(it->y - FACE_POS_OFFSET)) && (facesInfo[p].facePos.y < (it->y + FACE_POS_OFFSET)))
				break;
		}

		if (it == newFacePos.end())
		{
			++facesInfo[p].undetected_counter;
#ifdef SHOW_DEBUG_MESSAGES
			std::cout << "undetected: " << facesInfo[p].undetected_counter << std::endl;
#endif
		}

		if (facesInfo[p].undetected_counter > UNDETECTED_THREHOLD)
		{
			facesInfo.erase(facesInfo.begin() + p);
#ifdef SHOW_DEBUG_MESSAGES
			std::cout << "erase: " << p << std::endl;
#endif
		}
	}

	// evaluate a list of possible faces
	for (i = 0, it = newFacePos.begin(); it != newFacePos.end(); ++it, ++i)
	{
		++face_num;
		cv::Mat face = original(*it).clone();
		//resize(face, face, cv::Size(100, 100));
		cv::cvtColor(face, face_grey, CV_BGR2GRAY);
		//cout << "\t" << detector.hasEyes(face) << endl;

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
#endif

#ifdef COMPARE_FACE_COLOUR
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

		cv::Point center(it->x + it->width*0.5, it->y + it->height*0.5);
		cv::Point top(it->x, it->y);

#ifdef COMPARE_FACE_COLOUR
		if ((similar_pixel_counter > min_percent) && (similar_pixel_counter < max_percent))  // if the percentage of similar pixeel is within certain range, it is a face
#endif
		{
			model->predict(face_grey, predictedLabel, confidence);
			if (confidence > confidence_threshold)
				predictedLabel = Guest;

			isExistedFace = false;
			for (p = 0; p < facesInfo.size(); ++p)
			{
				//cout << facesInfo[p].facePos.x << "," << facesInfo[p].facePos.y << endl;
				if (!isExistedFace &&
					(facesInfo[p].facePos.x > (top.x - FACE_POS_OFFSET)) &&
					(facesInfo[p].facePos.x < (top.x + FACE_POS_OFFSET)) &&
					(facesInfo[p].facePos.y > (top.y - FACE_POS_OFFSET)) &&
					(facesInfo[p].facePos.y < (top.y + FACE_POS_OFFSET)))
				{
					facesInfo[p].facePos.x = top.x;
					facesInfo[p].facePos.y = top.y;

					++(facesInfo[p].face_counter[predictedLabel]);

					if (!(facesInfo[p].isRecognized))
					{
						std::string str;
						oss.str("");
						switch (predictedLabel)
						{
						case Joel:
						case KaHo:
						case Yumi:
							if (facesInfo[p].face_counter[predictedLabel] >= FACE_DET_THREHOLD)
							{
#ifdef SHOW_MARKERS
								//oss << PERSON_NAME[facesInfo[p].label] << " " << confidence;
								oss << PERSON_NAME[predictedLabel] << " detected";
#endif
								facesInfo[p].isRecognized = true;
								facesInfo[p].label = (DETECTED_PERSON)predictedLabel;
#ifdef SHOW_DEBUG_MESSAGES
								std::cout << "detected: " << predictedLabel << '\n';
#endif

								str = std::string(HELLO_MESSAGE) + std::string(PERSON_NAME[predictedLabel]);
								/* Text to Speech */
								TextToSpeech::pushBack(str);
								//tts.speakNow(str);
							}
#ifdef SHOW_MARKERS
							else {
								//oss << DETECTING << " " << confidence;
								//oss << DETECTING << ", maybe " << PERSON_NAME[facesInfo[p].label];
								//oss << DETECTING << ", maybe " << PERSON_NAME[predictedLabel] << "-" << confidence;
								oss << "maybe " << PERSON_NAME[predictedLabel] << "-" << confidence;
							}
#endif
							break;

						case Guest:
							if (facesInfo[p].face_counter[Guest] >= FACE_DET_THREHOLD * 2)
							{
#ifdef SHOW_MARKERS
								oss << PERSON_NAME[Guest] << " " << confidence;
								//oss << NAME_GUEST;
#endif
								if (facesInfo[p].face_counter[Guest] == 10)
								{
									str = std::string(HELLO_MESSAGE) + std::string(PERSON_NAME[Guest]);
									/* Text to Speech */
									TextToSpeech::pushBack(str);
									//tts.speakNow(str);
								}
							}
#ifdef SHOW_MARKERS
							else {
								oss << DETECTING << confidence;
								//oss << DETECTING;
							}
#endif
							break;

						case -1:
						default:
#ifdef SHOW_MARKERS
							oss << "unrecognised";
#endif
							break;
						}

					}
#ifdef SHOW_MARKERS
					else
					{
						oss.str("");
						switch (facesInfo[p].label)
						{
						case Guest:
						case Joel:
						case KaHo:
						case Yumi:
							//oss << PERSON_NAME[facesInfo[p].label] << "-" << confidence;
							oss << "D:" << PERSON_NAME[facesInfo[p].label] << ",R:" << PERSON_NAME[predictedLabel] << "-" << confidence;
							//oss << PERSON_NAME[facesInfo[p].label];
							break;

						default:
							oss << "Special!! " << confidence;
							//oss << NAME_GUEST;
							break;
						}
					}
#endif
#ifdef SHOW_MARKERS
					putText(*frame, oss.str(), top, cv::FONT_HERSHEY_SIMPLEX, 1,
						cv::Scalar(0, 0, 255), 2, 12);
#endif
					isExistedFace = true;
				}
				//cout << "face[" << i << "][" << p << "]: und=" << facesInfo[p].undetected_counter << endl;
			}

			if (facesInfo.size() == 0 || !isExistedFace)
			{
				DetectionInfo para;
				memset(&para, 0, sizeof(DetectionInfo));
				para.isRecognized = false;
				para.facePos = cv::Point(it->x, it->y);
				para.face_counter[predictedLabel] = 1;
				facesInfo.push_back(para);

#ifdef SHOW_MARKERS
				oss.str("");
				oss << "maybe " << PERSON_NAME[predictedLabel] << "-" << confidence;
				putText(*frame, oss.str(), top, cv::FONT_HERSHEY_SIMPLEX, 1,
					cv::Scalar(255, 0, 255), 2, 12);
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

#ifdef SAVE_IMAGES
	if (i > 0) {
		oss.str("");
		oss << IMAGE_DIR << image_num << IMAGE_NAME_POSTFIX << IMAGE_EXTENSION;
		imwrite(oss.str(), original);
		oss.str("");
		oss << IMAGE_DIR << "frame_" << image_num << IMAGE_NAME_POSTFIX << IMAGE_EXTENSION;
		imwrite(oss.str(), *frame);
		++image_num;
	}
#endif


#ifdef DISPLAY_IMAGES
	cv::namedWindow("Video_Stream", CV_WINDOW_AUTOSIZE);     // Create a window for display.
	cv::imshow("Video_Stream", *frame);
#endif

	return 0;
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
