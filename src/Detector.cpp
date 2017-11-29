#include "Detector.h"

Detector::Detector(const std::string shapePredictorFilePath)
{
	if (&shapePredictorFilePath == NULL || shapePredictorFilePath.empty()) {
		std::cerr << "Shape predictor file path is an empty string." << std::endl;
	}

	frontalFaceDetector = new dlib::frontal_face_detector(dlib::get_frontal_face_detector());
	shapePredictor = new dlib::shape_predictor();
	dlib::deserialize(shapePredictorFilePath) >> *shapePredictor;
	resizedWidth = RESIZED_WIDTH;
	flipFrame = FLIP_FRAME;
	searchPupil = SEARCH_PUPIL;
}

void Detector::SetResizedWidth(const int width)
{
	resizedWidth = std::max(width, 1);
}

int Detector::GetResizedWidth() const
{
	return resizedWidth;
}

void Detector::SetFlipFrame(const bool flip)
{
	flipFrame = flip;
}

bool Detector::GetFlipFrame() const
{
	return flipFrame;
}

void Detector::SetSearchPupil(const bool search)
{
	searchPupil = search;
}

bool Detector::GetSearchPupil() const
{
	return searchPupil;
}

bool Detector::IsFaceFound() const
{
	return faces.size() != 0;
}

std::vector<Face*> Detector::GetFaces() const
{
	return faces;
}

int Detector::GetTotalBlinks() const {
	int blinks = 0;
	for (size_t i = 0; i < faces.size(); i++)
	{
		blinks += faces.at(i)->GetBlinks();
	}
	return blinks;
}

Detector::~Detector()
{
	if (&frontalFaceDetector != NULL)
	{
		delete frontalFaceDetector;
	}
	if (&shapePredictor != NULL)
	{
		delete shapePredictor;
	}
	for each (Face* face in faces)
	{
		if (&face != NULL)
		{
			delete face;
		}
	}
	faces.clear();
}

void Detector::Detect(cv::Mat &frame)
{
	if (frame.empty()) {
		return;
	}

	try
	{
		// flip frame if frame is captured from front camera
		if (flipFrame) {
			cv::flip(frame, frame, 1);
		}

		if (countOfFrames % SKIP_FRAMES == 0)
		{
			// Downscale frame to resizedWidth width
			double scale = (double)std::min(resizedWidth, frame.cols) / frame.cols;
			cv::Size resizedFrameSize = cv::Size((int)(scale*frame.cols), (int)(scale*frame.rows));

			// resize the frame
			cv::Mat resizedFrame;
			cv::resize(frame, resizedFrame, resizedFrameSize, 0.0, 0.0, cv::INTER_AREA);
			// create grayscale and equalize the histogram
			cv::Mat gray1Channel;
			cv::cvtColor(resizedFrame, gray1Channel, CV_BGR2GRAY);
			cv::equalizeHist(gray1Channel, gray1Channel);
			// create 3 channel Mat from grayscale for dlib
			cv::Mat in[] = { gray1Channel, gray1Channel, gray1Channel };
			cv::Mat gray3Channels;
			cv::merge(in, 3, gray3Channels);

			// Change to dlib's image format. No memory is copied.
			dlib::cv_image<dlib::bgr_pixel> cvImageSmall(gray3Channels);

			// Detect faces on resize image
			std::vector<dlib::rectangle> foundFaces;
			foundFaces = (*frontalFaceDetector)(cvImageSmall, -0.5);
			for (size_t i = 0; i < foundFaces.size(); i++)
			{
				dlib::rectangle* foundFace = &foundFaces.at(i);
				Face* face;
				if (faces.size() <= i)
				{
					Face* newFace = new Face();
					faces.push_back(newFace);
				}
				face = faces.at(i);
				face->SetFace(Helpers::ScaleUp(*foundFace, scale));
			}
			for (size_t i = foundFaces.size(); i < faces.size(); i++)
			{
				Face* face = faces.at(i);
				face->Reset();
			}

			// Find the pose of each face.
			for (size_t i = 0; i < faces.size(); i++)
			{
				// Landmark detection on image
				Face* face = faces.at(i);
				if (face->HasFace())
				{
					// Set Landmarks
					face->SetShape(Helpers::ScaleUp((*shapePredictor)(cvImageSmall, Helpers::ScaleDown(face->GetFace(), scale)), scale));
					// Find Pupils
					if (searchPupil) {
						face->SetLeftPupil(Helpers::ScaleUp(Helpers::FindPupilCenter(gray1Channel,
							Helpers::ScaleDown(face->GetLeftEyeRect(), scale)), scale));
						face->SetRightPupil(Helpers::ScaleUp(Helpers::FindPupilCenter(gray1Channel,
							Helpers::ScaleDown(face->GetRightEyeRect(), scale)), scale));
					}
					// Check for blink
					face->CheckBlink();
				}
			}
		}
	}
	catch (const std::exception& ex)
	{
		printf(ex.what());
	}
	// increase counter
	countOfFrames++;
}

void Detector::operator >> (cv::Mat &frame)
{
	return this->Detect(frame);
}
