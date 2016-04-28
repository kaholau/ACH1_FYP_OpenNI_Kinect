#ifndef DETECTOR_H
#define DETECTOR_H

#include <iostream>
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#define NUMBER_OF_IMAGES   5
#define NUMBER_OF_CHANNELS 3
#define MIN_ITEM           0
#define MAX_ITEM           1

#define ORIGINAL_WIDTH     1280
#define ORIGINAL_HEIGHT    960

#define RESIZE_TO_SMALLER
#ifdef RESIZE_TO_SMALLER
	#define RESIZE_SCALE       (8.0/3.0)
	#define RESIZE_WIDTH       ((float)ORIGINAL_WIDTH / RESIZE_SCALE)
	#define RESIZE_HEIGHT      ((float)ORIGINAL_HEIGHT / RESIZE_SCALE)
	#define MIN_FACE_SIZE      (RESIZE_WIDTH / 24)
	#define MAX_FACE_SIZE      (RESIZE_WIDTH / 3.5)
#else
	#define MIN_FACE_SIZE      80
	#define MAX_FACE_SIZE      600
#endif

class Detector
{
public:
    typedef struct Attr
    {
        cv::Mat image;
        cv::vector<cv::Rect> objects;
        double scaleFactor;
        int minNeighbors;
        int flags;
		cv::Size minSize;
		cv::Size maxSize;
    } attr;

    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;

    Detector();

	void getFaces(const cv::Mat &image, cv::vector<cv::Rect>& faces_pos);
	bool hasEyes(cv::Mat &image);
	void compareFaceColour(cv::Mat &image, cv::Mat &outputMask);
    double* getFaceColourAvg(void);
    double* getFaceColourDev(void);
    double* getFaceColourDevConst(void);
    void setFaceColourDevConst(double value);
	cv::Mat resizeToSmaller(cv::Mat *);

private:
	cv::String face_cascade_name;
	cv::String eyes_cascade_name;

    attr attr_face;
    attr attr_eye;

    double faceColour_avg[NUMBER_OF_CHANNELS];
    double faceColour_dev[NUMBER_OF_CHANNELS];
    double faceColour_dev_const;
};

#endif // DETECTOR_H
