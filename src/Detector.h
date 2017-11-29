#pragma once

#include "Face.h"
#include "Constants.h"
#include "Helpers.h"

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <iostream>

#include <dlib\image_processing\frontal_face_detector.h>
#include <dlib\image_processing.h>
#include <dlib\opencv.h>

class Detector
{
public:
	Detector(const std::string shapePredictorFilePath);
	~Detector();

	std::vector<Face*>		GetFaces() const;
	void					Detect(cv::Mat &frame);
	void					operator >> (cv::Mat &frame);
	void                    SetResizedWidth(const int width);
	void					SetFlipFrame(const bool flip);
	void					SetSearchPupil(const bool search);
	int                     GetResizedWidth() const;
	int						GetTotalBlinks() const;
	bool					GetFlipFrame() const;
	bool					GetSearchPupil() const;
	bool					IsFaceFound() const;

private:
	dlib::shape_predictor*			shapePredictor = NULL;
	dlib::frontal_face_detector*	frontalFaceDetector = NULL;
	std::vector<Face*>				faces;
	int								resizedWidth = 0;
	int								countOfFrames = 0;
	bool							flipFrame;
	bool							searchPupil;
};
