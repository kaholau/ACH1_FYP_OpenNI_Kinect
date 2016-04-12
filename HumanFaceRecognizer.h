#pragma once

#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sstream>
#include <string.h>

#include "include.h"
//#include "detector.h"
//#include "TextToSpeech.h"


/* Defines */
//#define DURATION_CHECK_FACE
//#define SAVE_IMAGES
//#define SAVE_FACES
//#define SAVE_MASKS
//#define DISPLAY_FACES_AND_MASKS
//#define DISPLAY_IMAGES
#define SHOW_MARKERS
#define COMPARE_FACE_COLOUR

#define NUM_OF_PERSON		3

#define BASE_DIR           "detected_faces/"
#define IMAGE_DIR          "image/"
#define CORRECT_DIR        "correct/"
#define WRONG_DIR          "wrong/"
#define IMAGE_NAME_POSTFIX "_image"
#define FACE_NAME_POSTFIX  "_face"
#define MASK_NAME_POSTFIX  "_face_mask"
#define IMAGE_EXTENSION    ".bmp"

#define DB_FACE_FILE_PATH  "db/lbp_face.xml"
#define DB_NAME_FILE_PATH  "db/name.csv"

#define DETECTING          "detecting..."
#define HELLO_MESSAGE      "This is "

#define FACE_POS_OFFSET			40
#define FACE_DET_THREHOLD		4
#define UNDETECTED_THREHOLD		(FACE_DET_THREHOLD*1.5)
#define NUM_OF_CHANNELS_COLOUR	3


/* Enum */
typedef enum DETECTED_PERSON {
	Guest = 0,
	Joel = 1,
	KaHo = 2,
	Yumi = 3
} DETECTED_PERSON;


/* Global Variables */


/* Class */
typedef class HumanFaceRecognizer
{
public:
	bool isAddNewFace;
	bool isUpdated;
	cv::Ptr<cv::FaceRecognizer> model;
	std::vector<std::string> PERSON_NAME;


	double total_percent;
	double total_percent_var;
	unsigned int num_of_face_detected;

	HumanFaceRecognizer();
	~HumanFaceRecognizer();

	int runFaceRecognizer(cv::Mat *);

	void addNewFace(cv::Mat &frame, std::string name);

	void testExample(void);

private:
	Detector detector;
	double min_percent;
	double max_percent;
	int num_of_person_in_db;
	std::vector<struct DetectionInfo> facesInfo;

} HumanFaceRecognizer;
