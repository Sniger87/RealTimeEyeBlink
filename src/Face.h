#pragma once

#include "Constants.h"
#include "Helpers.h"

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <dlib\image_processing.h>

class Face
{
public:
	Face();
	~Face();

	dlib::rectangle				GetFace() const;
	cv::Rect					GetFaceRect() const;
	std::vector<cv::Point>		GetShape() const;
	cv::Rect					GetShapeRect() const;
	std::vector<cv::Point>		GetMouth() const;
	std::vector<cv::Point>		GetEyes() const;
	std::vector<cv::Point>		GetLeftEye() const;
	cv::Rect					GetLeftEyeRect() const;
	std::vector<cv::Point>		GetRightEye() const;
	cv::Rect					GetRightEyeRect() const;
	std::vector<cv::Point>		GetNose() const;
	std::vector<cv::Point>		GetEyebrows() const;
	std::vector<cv::Point>		GetJaw() const;
	cv::Point					GetLeftPupil() const;
	cv::Point					GetRightPupil() const;
	void						SetLeftPupil(const cv::Point point);
	void						SetRightPupil(const cv::Point point);
	void						SetFace(const dlib::rectangle faceRect);
	void						SetShape(const dlib::full_object_detection shapeDetect);
	void						CheckBlink();
	void						Reset();
	int							GetBlinks() const;
	bool						HasFace() const;

private:
	dlib::full_object_detection	shape;
	dlib::rectangle				face;
	cv::Point					leftPupil;
	cv::Point					rightPupil;

	float						blinkThreshold = 0.0f;
	int							blinks = 0;
	int							counterFrames = 0;
	bool						hasFace = false;
};
