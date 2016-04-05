#include "detector.h"

//#define LOCAL_FACE_COLOUR
//#define FACE_MASK_COLOUR


Detector::Detector()
{

	// Below statistics were obtained by evaluating the faces of LFW database
	faceColour_avg[0] = 100.175;    // B
	faceColour_avg[1] = 120.392;    // G
	faceColour_avg[2] = 153.315;    // R

	faceColour_dev_const = 1.15;    // this could be changed
	faceColour_dev[0] = 68.3169;    // B
	faceColour_dev[1] = 72.5205;    // G
	faceColour_dev[2] = 76.6897;    // R

    attr_face.scaleFactor = 1.08;
    attr_face.minNeighbors = 6;
    attr_face.flags = 0|CV_HAAR_SCALE_IMAGE;
	attr_face.minSize.width = attr_face.minSize.height = MIN_FACE_SIZE;
	attr_face.maxSize.width = attr_face.maxSize.height = MAX_FACE_SIZE;

    attr_eye.scaleFactor = 1.05;
    attr_eye.minNeighbors = 3;
    attr_eye.flags = 0|CV_HAAR_SCALE_IMAGE;
    attr_eye.minSize.width = 6;
    attr_eye.minSize.height = 3;


    // Load the cascades
    face_cascade_name = "db/haarcascade_frontalface_alt.xml";
    eyes_cascade_name = "db/haarcascade_eye_tree_eyeglasses.xml";
    if( !face_cascade.load(face_cascade_name) )
		std::cout << "Face Features loading error!\n";

    if( !eyes_cascade.load( eyes_cascade_name ) )
		std::cout << "Eye Features loading error!\n";
}

void Detector::getFaces(const cv::Mat &image, cv::vector<cv::Rect> &faces_pos)
{
	cv::Mat image_gray;

    cvtColor( image, image_gray, CV_BGR2GRAY );
	cv::equalizeHist(image_gray, image_gray);

    // Detect faces
    face_cascade.detectMultiScale(image_gray, faces_pos, attr_face.scaleFactor,
		attr_face.minNeighbors, attr_face.flags, attr_face.minSize );

	return ;
}

bool Detector::hasEyes(cv::Mat &image)
{
	cv::Mat face_gray;
	std::vector<cv::Rect> eyes;

    cvtColor( image, face_gray, CV_BGR2GRAY );
    equalizeHist( face_gray, face_gray );

    //-- In each face, detect eyes
	eyes_cascade.detectMultiScale(face_gray, eyes, attr_eye.scaleFactor, attr_eye.minNeighbors, attr_eye.flags, attr_eye.minSize, attr_eye.maxSize);

    // Draw circles on the eyes
//    for( size_t j = 0; j < eyes.size(); j++ )
//    {
//        Point center( eyes[j].x + eyes[j].width*0.5, eyes[j].y + eyes[j].height*0.5 );
//        int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
//        circle( image, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
//    }

    if (eyes.size() == 2)
        return true;

    return false;
}

void Detector::compareFaceColour( cv::Mat &image, cv::Mat &outputMask)
{
    int j, k, l;

#ifdef LOCAL_FACE_COLOUR
    int rows = 0, cols = 0;
    int face_pixel = 0;
    double face_color_avg[NUMBER_OF_CHANNELS] = { 0 };

    // Calculate the average colour of the faces
    double face_color_dev[NUMBER_OF_CHANNELS] = { 0 };
    double face_color_var[NUMBER_OF_CHANNELS] = { 0 };
    Mat tmpMat = image;
    face_pixel = tmpMat.rows * tmpMat.cols; // find out the number of pixels of a face

    for(j=0; j<NUMBER_OF_CHANNELS; ++j)
    {
        face_color_avg[j] = cv::sum(tmpMat)[j]; // sum all the pixels of each channel of a face
        face_color_avg[j] /= face_pixel;
    }

    cout << "face [ ] = " << face_color_avg[0] << ",\t" << face_color_avg[1] << ",\t" << face_color_avg[2] << "\t|" << face_pixel << endl;

    // Calculate the variance and standard deviation of the faces
    for(j=0; j < tmpMat.rows; ++j)
    {
        for(k=0; k < tmpMat.cols; ++k)
        {
            for(l=0; l < NUMBER_OF_CHANNELS; ++l)
            {
                face_color_var[l] += pow( (*(tmpMat.data + ((j*tmpMat.cols+k) * NUMBER_OF_CHANNELS + l)) - face_color_avg[l]), 2 );
            }
        }
    }
#endif

    double threshold[NUMBER_OF_CHANNELS][2];

    for(l=0; l < NUMBER_OF_CHANNELS; ++l)
    {
#ifdef LOCAL_FACE_COLOUR
        face_color_var[l] /= face_pixel;
        face_color_dev[l] = sqrt( face_color_var[l] );
#endif

        // Calculate the threshold
		threshold[l][MIN_ITEM] = faceColour_avg[l] - faceColour_dev_const * faceColour_dev[l];
		threshold[l][MAX_ITEM] = faceColour_avg[l] + faceColour_dev_const * faceColour_dev[l];
    }


    // change the faces into black and white
    outputMask = image.clone();

    for(j=0; j < outputMask.rows; ++j)
    {
        for(k=0; k < outputMask.cols; ++k)
        {
            for(l=0; l < NUMBER_OF_CHANNELS; ++l)
            {
				if ((*(outputMask.data + ((j*outputMask.cols + k) * NUMBER_OF_CHANNELS + l)) >= threshold[l][MIN_ITEM]) &&
					(*(outputMask.data + ((j*outputMask.cols + k) * NUMBER_OF_CHANNELS + l)) <= threshold[l][MAX_ITEM]))
                {
                    *(outputMask.data + ((j*outputMask.cols+k) * NUMBER_OF_CHANNELS + l)) = 255;
//                    *(outputMask.data + ((j*outputMask.cols+k) * NUMBER_OF_CHANNELS + 0)) = 255;
//                    *(outputMask.data + ((j*outputMask.cols+k) * NUMBER_OF_CHANNELS + 1)) = 255;
//                    *(outputMask.data + ((j*outputMask.cols+k) * NUMBER_OF_CHANNELS + 2)) = 255;
                } else {
                    *(outputMask.data + ((j*outputMask.cols+k) * NUMBER_OF_CHANNELS + l)) = 0;
//                    *(outputMask.data + ((j*outputMask.cols+k) * NUMBER_OF_CHANNELS + 0)) = 0;
//                    *(outputMask.data + ((j*outputMask.cols+k) * NUMBER_OF_CHANNELS + 1)) = 0;
//                    *(outputMask.data + ((j*outputMask.cols+k) * NUMBER_OF_CHANNELS + 2)) = 0;
//                    l = 2;
                }
            }
        }
	}
#ifndef FACE_MASK_COLOUR
    cvtColor( outputMask, outputMask, CV_BGR2GRAY );
#endif

    return ;
}

double* Detector::getFaceColourAvg( void )
{
    return faceColour_avg;
}

double* Detector::getFaceColourDev( void )
{
    return faceColour_dev;
}

double* Detector::getFaceColourDevConst( void )
{
    return &faceColour_dev_const;
}

void Detector::setFaceColourDevConst( double value )
{
    faceColour_dev_const = value;
    return ;
}

#ifdef RESIZE_TO_SMALLER
cv::Mat Detector::resizeToSmaller(cv::Mat *frame)
{
	if (frame->size().width == 640 && frame->size().height == 480)
		return (*frame).clone();

	cv::Mat in = (*frame).clone();
	const cv::Size size(RESIZE_WIDTH, RESIZE_HEIGHT);
	cv::resize(*frame, *frame, size);

	return in;
}
#endif

