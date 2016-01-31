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
    double* getFaceColourAvg( void );
    double* getFaceColourDev( void );
    double* getFaceColourDevConst( void );
    void setFaceColourDevConst( double value );

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
