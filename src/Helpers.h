#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <dlib\image_processing.h>

class Helpers
{
public:
	static cv::Rect ScaleDown(const cv::Rect &rect, const double scale) {
		return cv::Rect(rect.x * scale, rect.y * scale, rect.width * scale, rect.height * scale);
	}

	static cv::Rect ScaleUp(const cv::Rect &rect, const double scale) {
		return cv::Rect(rect.x / scale, rect.y / scale, rect.width / scale, rect.height / scale);
	}

	static dlib::rectangle ScaleDown(const dlib::rectangle &rectangle, const double scale) {
		return dlib::rectangle(rectangle.left() * scale, rectangle.top() * scale, rectangle.right() * scale, rectangle.bottom() * scale);
	}

	static dlib::rectangle ScaleUp(const dlib::rectangle &rectangle, const double scale) {
		return dlib::rectangle(rectangle.left() / scale, rectangle.top() / scale, rectangle.right() / scale, rectangle.bottom() / scale);
	}

	static dlib::full_object_detection ScaleDown(const dlib::full_object_detection &shapeDetect, const double scale) {
		std::vector<dlib::point> numParts;
		for (size_t i = 0; i < shapeDetect.num_parts(); i++)
		{
			numParts.push_back(dlib::point(shapeDetect.part(i).x() * scale, shapeDetect.part(i).y() * scale));
		}
		dlib::full_object_detection detection(
			dlib::rectangle(
				shapeDetect.get_rect().left() * scale,
				shapeDetect.get_rect().top() * scale,
				shapeDetect.get_rect().right() * scale,
				shapeDetect.get_rect().bottom() * scale), numParts);
		return detection;
	}

	static dlib::full_object_detection ScaleUp(const dlib::full_object_detection &shapeDetect, const double scale) {
		std::vector<dlib::point> numParts;
		for (size_t i = 0; i < shapeDetect.num_parts(); i++)
		{
			numParts.push_back(dlib::point(shapeDetect.part(i).x() / scale, shapeDetect.part(i).y() / scale));
		}
		dlib::full_object_detection detection(
			dlib::rectangle(
				shapeDetect.get_rect().left() / scale,
				shapeDetect.get_rect().top() / scale,
				shapeDetect.get_rect().right() / scale,
				shapeDetect.get_rect().bottom() / scale), numParts);
		return detection;
	}

	static cv::Point ScaleDown(const cv::Point &point, const double scale) {
		return cv::Point(point.x * scale, point.y * scale);
	}

	static cv::Point ScaleUp(const cv::Point &point, const double scale) {
		return cv::Point(point.x / scale, point.y / scale);
	}

	static bool IsRectInMat(const cv::Rect &rect, const cv::Mat &mat) {
		return rect.x > 0 && rect.y > 0 && rect.x + rect.width < mat.cols &&
			rect.y + rect.height < mat.rows;
	}

	static bool IsPointInMat(const cv::Point &p, const cv::Mat &mat) {
		return p.x >= 0 && p.x < mat.cols && p.y >= 0 && p.y < mat.rows;
	}

	static cv::Rect ConvertRectangleToRect(const dlib::rectangle &rectangle) {
		return cv::Rect(rectangle.left(), rectangle.top(), rectangle.width(), rectangle.height());
	}

	static cv::Point ScalePointToSize(const cv::Point p, const cv::Rect originalSize, const int width) {
		float ratio = (((float)width) / originalSize.width);
		int x = round(p.x / ratio);
		int y = round(p.y / ratio);
		return cv::Point(x, y);
	}

	static void ScaleMatToWidth(const cv::Mat &src, cv::Mat &dst, const int width) {
		cv::resize(src, dst, cv::Size(width, (((float)width) / src.cols) * src.rows));
	}

	static cv::Mat GetMatrixMagnitude(const cv::Mat &matX, const cv::Mat &matY) {
		cv::Mat mags(matX.rows, matX.cols, CV_64F);
		for (int y = 0; y < matX.rows; ++y) {
			const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
			double *Mr = mags.ptr<double>(y);
			for (int x = 0; x < matX.cols; ++x) {
				double gX = Xr[x], gY = Yr[x];
				double magnitude = sqrt((gX * gX) + (gY * gY));
				Mr[x] = magnitude;
			}
		}
		return mags;
	}

	static double GetGradientThreshold(const cv::Mat &mat, const double stdDevFactor) {
		cv::Scalar stdMagnGrad, meanMagnGrad;
		cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
		double stdDev = stdMagnGrad[0] / sqrt(mat.rows * mat.cols);
		return stdDevFactor * stdDev + meanMagnGrad[0];
	}

	static float GetEyeAspectRatio(const std::vector<cv::Point> points) {
		if (points.size() != 6)
		{
			return 1.0;
		}
		float a = cv::norm(points.at(1) - points.at(5));
		float b = cv::norm(points.at(2) - points.at(4));
		float c = cv::norm(points.at(0) - points.at(3));
		float ear = (a + b) / (2.0f * c);
		return ear;
	}

	static cv::Mat GetMatXGradient(const cv::Mat &mat) {
		cv::Mat out(mat.rows, mat.cols, CV_64F);
		for (int y = 0; y < mat.rows; ++y) {
			const uchar *Mr = mat.ptr<uchar>(y);
			double *Or = out.ptr<double>(y);

			Or[0] = Mr[1] - Mr[0];
			for (int x = 1; x < mat.cols - 1; ++x) {
				Or[x] = (Mr[x + 1] - Mr[x - 1]) / 2.0;
			}
			Or[mat.cols - 1] = Mr[mat.cols - 1] - Mr[mat.cols - 2];
		}
		return out;
	}

	static void GetPossibleCenters(int x, int y, const cv::Mat &weight, double gx, double gy, cv::Mat &out) {
		// for all possible centers
		for (int cy = 0; cy < out.rows; ++cy) {
			double *Or = out.ptr<double>(cy);
			const unsigned char *Wr = weight.ptr<unsigned char>(cy);
			for (int cx = 0; cx < out.cols; ++cx) {
				if (x == cx && y == cy) {
					continue;
				}
				// create a vector from the possible center to the gradient origin
				double dx = x - cx;
				double dy = y - cy;
				// normalize d
				double magnitude = sqrt((dx * dx) + (dy * dy));
				dx = dx / magnitude;
				dy = dy / magnitude;
				double dotProduct = dx * gx + dy * gy;
				dotProduct = std::max(0.0, dotProduct);
				// square and multiply by the weight
				Or[cx] += dotProduct * dotProduct * (Wr[cx] / WEIGHT_DIVISOR);
			}
		}
	}

	static cv::Point FindPupilCenter(const cv::Mat &frameInGray, const cv::Rect &eyeRect) {
		if (!Helpers::IsRectInMat(eyeRect, frameInGray))
		{
			return cv::Point();
		}

		// Extract eye region from frame
		cv::Mat eyeRoi = frameInGray(eyeRect);
		// Resize region to best fast size
		Helpers::ScaleMatToWidth(eyeRoi, eyeRoi, MAT_WIDTH_FOR_PUPIL);

		// Find the gradient
		cv::Mat gradientX = Helpers::GetMatXGradient(eyeRoi);
		cv::Mat gradientY = Helpers::GetMatXGradient(eyeRoi.t()).t();
		// Normalize and threshold the gradient
		// compute all the magnitudes
		cv::Mat mags = Helpers::GetMatrixMagnitude(gradientX, gradientY);
		// compute the threshold
		double gradientThresh = Helpers::GetGradientThreshold(mags, GRADIENT_THRESHOLD);
		// normalize
		for (int y = 0; y < eyeRoi.rows; ++y) {
			double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
			const double *Mr = mags.ptr<double>(y);
			for (int x = 0; x < eyeRoi.cols; ++x) {
				double gX = Xr[x], gY = Yr[x];
				double magnitude = Mr[x];
				if (magnitude > gradientThresh) {
					Xr[x] = gX / magnitude;
					Yr[x] = gY / magnitude;
				}
				else {
					Xr[x] = 0.0;
					Yr[x] = 0.0;
				}
			}
		}
		// Create a blurred and inverted image for weighting
		cv::Mat weight;
		cv::GaussianBlur(eyeRoi, weight, cv::Size(WEIGHT_BLUR_SIZE, WEIGHT_BLUR_SIZE), 0, 0);
		for (int y = 0; y < weight.rows; ++y) {
			unsigned char *row = weight.ptr<unsigned char>(y);
			for (int x = 0; x < weight.cols; ++x) {
				row[x] = (255 - row[x]);
			}
		}
		// Run the algorithm
		cv::Mat outSum = cv::Mat::zeros(eyeRoi.rows, eyeRoi.cols, CV_64F);
		// for each possible gradient location
		// Note: these loops are reversed from the way the paper does them
		// it evaluates every possible center for each gradient location instead of
		// every possible gradient location for every center.
		for (int y = 0; y < weight.rows; ++y) {
			const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
			for (int x = 0; x < weight.cols; ++x) {
				double gX = Xr[x], gY = Yr[x];
				if (gX == 0.0 && gY == 0.0) {
					continue;
				}
				Helpers::GetPossibleCenters(x, y, weight, gX, gY, outSum);
			}
		}
		// scale all the values down, basically averaging them
		double numGradients = (weight.rows * weight.cols);
		cv::Mat out;
		outSum.convertTo(out, CV_32F, 1.0 / numGradients);
		// Find the maximum point
		cv::Point maxP;
		cv::minMaxLoc(out, NULL, NULL, NULL, &maxP);
		// Resize Point to original
		maxP = Helpers::ScalePointToSize(maxP, eyeRect, MAT_WIDTH_FOR_PUPIL);
		maxP.x += eyeRect.x;
		maxP.y += eyeRect.y;
		return maxP;
	}
};
