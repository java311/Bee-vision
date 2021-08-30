// thread example
#include <iostream>       // std::cout
#include <stdio.h>
#include <stack>

using namespace cv;
using namespace std;

class cameraThread {
private:
    int resX;
    int resY;
    int cameraID;
    bool stopFlag;
    VideoCapture cap;
    Mat frame;
    stack<Mat> frameStack;

public:

    cameraThread(int resX, int resY, int cameraID) {
        this->stopFlag = false;
        this->resX = resX;
        this->resY = resY;
        this->cameraID = cameraID;

        cap.open(cameraID, cv::CAP_ANY);

        //inti the frame queue

        // Check if camera opened successfully
        if (!cap.isOpened()) {
            cerr << "ERROR! Unable to open camera\n";
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, resX);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, resY);
    }

    Mat getFrame() {
        if (!frameStack.empty()) {
            return frameStack.top();
        }
        else {
            std::cout << "vacio" << std::endl;
            Mat1b m;
            return m;
        }
    }

    void stop() {
        stopFlag = true;
        cap.release();
        cout << "Stopping CPU camera thread..." << std::endl;
    }

    void camLoop(int x) 
    {
        while (stopFlag == false)  {
            // wait for a new frame from camera and store it into 'frame'
            cap.read(frame);
            // check if we succeeded
            if (frame.empty()) {
                cerr << "ERROR! blank frame grabbed\n";
                break;
            }
            frameStack.push(frame);

            // show live and wait for a key with timeout long enough to show images
            //imshow("Live", frame);
            //if (waitKey(5) >= 0)
            //    break;
        }
    }

};




#pragma once
