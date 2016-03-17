#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sstream>
#include <string.h>

#include "detector.h"
#include "TextToSpeech.h"


/* Defines */
#define SAVE_IMAGES
#define SAVE_FACES
#define SAVE_MASKS
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
#define IMAGE_EXTENSION    ".jpg"

#define DB_FILE_PATH       "db/lbp_face.xml"

#define DETECTING          "detecting..."
#define HELLO_MESSAGE      "Hello, "

#define MAX_FACE_SIZE			600
#define FACE_POS_OFFSET			100
#define FACE_DET_THREHOLD		3
#define UNDETECTED_THREHOLD		(FACE_DET_THREHOLD*2)
#define NUM_OF_CHANNELS_COLOUR	3


/* Enum */
typedef enum DETECTED_PERSON {
	Guest = 0,
	Joel = 1,
	KaHo = 2,
	Yumi = 3
} DETECTED_PERSON;

/* Struct */
typedef struct DetectionInfo
{
	bool isRecognized;
	DETECTED_PERSON label;
	cv::Point facePos;
	short face_counter[NUM_OF_PERSON + 1];
	short undetected_counter;
} DetectionInfo;


/* Global Variables */
extern const std::string PERSON_NAME[NUM_OF_PERSON + 1];


/* Class */
typedef class HumanFaceRecognizer
{
public:
	double total_percent;
	double total_percent_var;
	unsigned int num_of_face_detected;

	HumanFaceRecognizer();
	~HumanFaceRecognizer();

	int runFaceRecognizer(cv::Mat *);

	void testExample(void);

private:
	Detector detector;
	double min_percent;
	double max_percent;
	cv::Ptr<cv::FaceRecognizer> model;
	std::vector<DetectionInfo> facesInfo;
} HumanFaceRecognizer;