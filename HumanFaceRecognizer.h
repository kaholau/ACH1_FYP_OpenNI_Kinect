#pragma once

#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sstream>
#include <string.h>
#include "include.h"


/* Defines */
//#define DURATION_CHECK_FACE
//#define SAVE_IMAGES
//#define SAVE_FACES
//#define SAVE_MASKS
//#define DISPLAY_FACES_AND_MASKS
//#define DISPLAY_IMAGES
#define COMPARE_FACE_COLOUR

#define NUM_OF_PERSON		3

#define BASE_DIR           "detected_faces/"
#define IMAGE_DIR          "image/"
#define CORRECT_DIR        "correct/"
#define WRONG_DIR          "wrong/"
#define IMAGE_NAME_POSTFIX "_image"
#define FACE_NAME_POSTFIX  "_face"
#define MASK_NAME_POSTFIX  "_face_mask"
#define IMAGE_EXTENSION    ".png"

#define DB_FACE_FILE_PATH  "db/lbp_face.xml"
#define DB_NAME_FILE_PATH  "db/name.csv"

#define DETECTING          "detecting..."
#define HELLO_MESSAGE      "This is "

#define FACE_REC_SIZE			120
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
	// Testing Use
	double totalConfidence;
	double total_percent;
	double total_percent_var;
	unsigned int num_of_face_detected;


	HumanFaceRecognizer();
	~HumanFaceRecognizer();

	int runFaceRecognizer(cv::Mat *);

	void addFace(cv::Mat &frame);

	void testExample(void);

	void clearNameStr();

	bool getisAddFace();

	bool getisUpdated();

	void saveFaceDatabase();

	void setisAddFace(bool b);

	void setisUpdated(bool b);

	void setNameStr(std::string name);

private:
	bool isAddFace;
	bool isUpdated;

	Detector detector;
	double min_percent;
	double max_percent;
	int num_of_person_in_db;

	cv::Ptr<cv::FaceRecognizer> model;
	std::vector<std::string> PERSON_NAME;
	std::vector<struct DetectionInfo> facesInfo;

	std::string NameStr;

	std::ofstream fout;


	void removeFaceWithClosedPos(void);
} HumanFaceRecognizer;
