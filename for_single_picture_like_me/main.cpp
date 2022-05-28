#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
#define ERROR cout << "Error on line " << __LINE__ << endl; system("pause");

String face_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml";
CascadeClassifier face_cascade;
String window_name = "result";

void process(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 5, CV_HAAR_SCALE_IMAGE, Size(30, 30));
    for (size_t i = 0; i < faces.size(); i++)
    {
        Point center;
        Mat faceROI = frame_gray(faces[i]);
        String cover = "face2.png"; // your "mask" goes here
        Mat signal = imread(cover), mask = imread(cover, 0);
        size_t w, h;
        w = faces[i].width;
        h = faces[i].height;
        size_t posx, posy;
        size_t k = 7;
        posx = std::max(1, faces[i].x);
        posy = std::max(1, faces[i].y);
        resize(signal, signal, Size(w, h), INTER_LINEAR);
        resize(mask, mask, Size(w, h), INTER_LINEAR);
        Rect rect(0, 0, std::min(w, frame.rows - posx - 1), std::min(h, frame.cols - posy - 1));
        w = std::min(w, frame.rows - posx - 1);
        h = std::min(h, frame.cols - posy - 1);
        Mat imageROI = frame(Rect(posx, posy, w, h));
        
        signal.copyTo(imageROI, mask);
    }
    imshow("Face Detection", frame);
    waitKey(0);
}

int main(int argc, char *argv[])
{ 
    cout << "wait for it" << endl;
    if (!face_cascade.load(face_cascade_name))
        ERROR;
    
    Mat frame;
    String Target = "123.jpeg";
    frame = imread(Target);
    if (frame.empty())
        ERROR;
    process(frame);
    return 0;
}
