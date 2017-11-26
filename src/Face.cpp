#include "Face.h"

Face::Face()
{
	blinkThreshold = EAR_BLINK_THRESHOLD;
}

Face::~Face()
{
}

dlib::rectangle Face::GetFace() const
{
	return face;
}

cv::Rect Face::GetFaceRect() const
{
	return Helpers::ConvertRectangleToRect(face);
}

std::vector<cv::Point> Face::GetShape() const
{
	std::vector<cv::Point> outShape;
	for (size_t i = 0; i < shape.num_parts(); i++)
	{
		outShape.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
	}
	return outShape;
}

cv::Rect Face::GetShapeRect() const {
	return Helpers::ConvertRectangleToRect(shape.get_rect());
}

std::vector<cv::Point> Face::GetJaw() const
{
	std::vector<cv::Point> outShape;
	if (shape.num_parts() >= 17)
	{
		for (size_t i = 0; i < 17; i++)
		{
			outShape.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
		}
	}
	return outShape;
}

std::vector<cv::Point> Face::GetEyebrows() const
{
	std::vector<cv::Point> outShape;
	if (shape.num_parts() >= 27)
	{
		for (size_t i = 17; i < 27; i++)
		{
			outShape.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
		}
	}
	return outShape;
}

std::vector<cv::Point> Face::GetNose() const
{
	std::vector<cv::Point> outShape;
	if (shape.num_parts() >= 36)
	{
		for (size_t i = 27; i < 36; i++)
		{
			outShape.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
		}
	}
	return outShape;
}

std::vector<cv::Point> Face::GetEyes() const
{
	std::vector<cv::Point> outShape;
	if (shape.num_parts() >= 48)
	{
		for (size_t i = 36; i < 48; i++)
		{
			outShape.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
		}
	}
	return outShape;
}

std::vector<cv::Point> Face::GetLeftEye() const
{
	std::vector<cv::Point> outShape;
	if (shape.num_parts() >= 42)
	{
		for (size_t i = 36; i < 42; i++)
		{
			outShape.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
		}
	}
	return outShape;
}

cv::Rect Face::GetLeftEyeRect() const {
	std::vector<cv::Point> eyePoints = GetLeftEye();
	if (eyePoints.size() != 6)
	{
		return cv::Rect();
	}
	cv::Rect result(eyePoints.at(0).x, eyePoints.at(1).y,
		(eyePoints.at(3).x - eyePoints.at(0).x), (eyePoints.at(4).y - eyePoints.at(1).y));
	result -= cv::Point(5, 5);
	result += cv::Size(10, 10);
	return result;
}

std::vector<cv::Point> Face::GetRightEye() const
{
	std::vector<cv::Point> outShape;
	if (shape.num_parts() >= 48)
	{
		for (size_t i = 42; i < 48; i++)
		{
			outShape.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
		}
	}
	return outShape;
}

cv::Rect Face::GetRightEyeRect() const {
	std::vector<cv::Point> eyePoints = GetRightEye();
	if (eyePoints.size() != 6)
	{
		return cv::Rect();
	}
	cv::Rect result(eyePoints.at(0).x, eyePoints.at(1).y,
		(eyePoints.at(3).x - eyePoints.at(0).x), (eyePoints.at(4).y - eyePoints.at(1).y));
	result -= cv::Point(5, 5);
	result += cv::Size(10, 10);
	return result;
}

std::vector<cv::Point> Face::GetMouth() const
{
	std::vector<cv::Point> outShape;
	if (shape.num_parts() >= 68)
	{
		for (size_t i = 48; i < 68; i++)
		{
			outShape.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
		}
	}
	return outShape;
}

void Face::SetFace(const dlib::rectangle faceRect)
{
	if (faceRect.height() > 0 && faceRect.width() > 0)
	{
		hasFace = true;
	}
	face = faceRect;
}

void Face::SetShape(const dlib::full_object_detection shapeDetect)
{
	shape = shapeDetect;
}

int Face::GetBlinks() const {
	return blinks;
}

void Face::Reset() {
	hasFace = false;
	shape = dlib::full_object_detection();
	leftPupil = cv::Point();
	rightPupil = cv::Point();
	face = dlib::rectangle();
	counterFrames = 0;
}

bool Face::HasFace() const {
	return hasFace;
}

cv::Point Face::GetLeftPupil() const {
	return leftPupil;
}

cv::Point Face::GetRightPupil() const {
	return rightPupil;
}

void Face::SetLeftPupil(const cv::Point point) {
	leftPupil = point;
}

void Face::SetRightPupil(const cv::Point point) {
	rightPupil = point;
}

void Face::CheckBlink() {
	float leftEar = Helpers::GetEyeAspectRatio(GetLeftEye());
	float rightEar = Helpers::GetEyeAspectRatio(GetRightEye());

	if (leftEar <= blinkThreshold && rightEar <= blinkThreshold)
	{
		counterFrames++;
	}
	else
	{
		counterFrames = 0;
	}

	if (counterFrames == MAX_EAR_FRAMES)
	{
		blinks++;
	}
}