// HMD-C-Port.cpp : This project is the CUDA/HMD version of the Bee Vision project v2.0
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>

#include <librealsense2/rs.hpp>
#include <windows.h>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <vector>

#include "cameraThread.h"
#include "cuStream.h"

using namespace cv;
using namespace std;

// #####################  GLOBAL VARIABLES 
int resX = 1280;
int resY = 720;
int RS_QUEUE_CAPACITY = 3;
float Rtmul = 1.0;
float Tdiv = 1000.0f;

// ############  INIT VARIABLES AND FUNCTIONS ############################
// Real Sense Control Objects
rs2::pipeline pipe;
rs2::config cfg;
float clipping_distance;

// LEFT AND RIGHT MAPPING MATRICES
vector<Mat> RC_Mats;
vector<Mat> LC_Mats;
Mat mapx1_right(cv::Size(resX, resY), CV_32FC1); Mat mapy1_right(cv::Size(resX, resY), CV_32FC1);
Mat mapx2_right(cv::Size(resX, resY), CV_32FC1); Mat mapy2_right(cv::Size(resX, resY), CV_32FC1);
Mat mapx1_left(cv::Size(resX, resY), CV_32FC1); Mat mapy1_left(cv::Size(resX, resY), CV_32FC1);
Mat mapx2_left(cv::Size(resX, resY), CV_32FC1); Mat mapy2_left(cv::Size(resX, resY), CV_32FC1);
// LEFT AND RIGHT BASE DISTANCE VARIABLES
double base_rgb_right;
double base_rgb_left;
// LEFT AND RIGHT PROJECTION VARIABLES AND MATRICES
vector<Mat> RSL_R_Mats;
Mat right_color_cam_matrix; Mat right_color_distort;
Mat Rt_right = cv::Mat::zeros(cv::Size(4, 4), CV_64F);
vector<Mat> RSL_L_Mats;
Mat left_color_cam_matrix; Mat left_color_distort;
Mat Rt_left = cv::Mat::zeros(cv::Size(4, 4), CV_64F);

// MATRICES FOR THE DEPTH DATA AND REGISTERED DEPTH DATA 
Mat depth_cam_matrix = cv::Mat::zeros(cv::Size(3, 3), CV_64F);
Mat depth_image_left;
Mat depth_image_right;

// ######## LOADS CAMERA MATRICES PARAMETERS FROM YML FILES ############
vector<Mat> loadCamFiles(string fExt, string fInt) 
{
    FileStorage fext(fExt, FileStorage::READ);
    FileStorage fint(fInt, FileStorage::READ);
    if (!fext.isOpened() || !fint.isOpened()) {
        cerr << "ERROR : Failed to open file : " << fExt << endl;
        vector<Mat> retMats;
        return retMats;
    }

    Mat R(cv::Size(3, 3), CV_32F);
    Mat T(cv::Size(1, 3), CV_32F);
    Mat R1(cv::Size(3, 3), CV_32F);
    Mat R2(cv::Size(3, 3), CV_32F);
    Mat P1(cv::Size(4, 3), CV_32F);
    Mat P2(cv::Size(4, 3), CV_32F);
    Mat Q(cv::Size(4, 4), CV_32F);
    fext["R"] >> R;
    fext["T"] >> T;
    fext["R1"] >> R1;
    fext["R2"] >> R2;
    fext["P1"] >> P1;
    fext["P2"] >> P2;
    fext["Q"] >> Q;

    Mat M1(cv::Size(3, 3), CV_32F);
    Mat D1(cv::Size(14, 1), CV_32F);
    Mat M2(cv::Size(3, 3), CV_32F);
    Mat D2(cv::Size(14, 1), CV_32F);
    fint["M1"] >> M1;
    fint["D1"] >> D1;
    fint["M2"] >> M2;
    fint["D2"] >> D2;

    //M1, M2, D1, D2, R1, R2, P1, P2, T, R
    vector<Mat> retMats;
    retMats.push_back(M1); retMats.push_back(M2);
    retMats.push_back(D1); retMats.push_back(D2);
    retMats.push_back(R1); retMats.push_back(R2);
    retMats.push_back(P1); retMats.push_back(P2);
    retMats.push_back(T); retMats.push_back(R);

    fint.release();
    fext.release();
    return retMats;
}

void initVariables() {
    RC_Mats = loadCamFiles("cam-params/right_center_extrinsics.yml", "cam-params/right_center_intrinsics.yml");
    LC_Mats = loadCamFiles("cam-params/left_center_extrinsics.yml", "cam-params/left_center_intrinsics.yml");

    // ### CALCULATE RIGHT CAMERA DISTORITON MATRICES
    cv::initUndistortRectifyMap(RC_Mats[0], RC_Mats[2], RC_Mats[4], RC_Mats[6], cv::Size(resX, resY), CV_32FC1, mapx1_right, mapy1_right);
    cv::initUndistortRectifyMap(RC_Mats[1], RC_Mats[3], RC_Mats[5], RC_Mats[7], cv::Size(resX, resY), CV_32FC1, mapx2_right, mapy2_right);
    // ### CALCULATE LEFT CAMERA DISTORTION MATRICES
    cv::initUndistortRectifyMap(LC_Mats[0], LC_Mats[2], LC_Mats[4], LC_Mats[6], cv::Size(resX, resY), CV_32FC1, mapx1_left, mapy1_left);
    cv::initUndistortRectifyMap(LC_Mats[1], LC_Mats[3], LC_Mats[5], LC_Mats[7], cv::Size(resX, resY), CV_32FC1, mapx2_left, mapy2_left);
    // #### CALCULATE THE BASE DISTANCE BETWEEN THE SENSOR AND THE CAMERAS 
    base_rgb_right = RC_Mats[8].at<double>(0) * RC_Mats[0].at<double>(0,0);  
    base_rgb_left  = LC_Mats[8].at<double>(0) * LC_Mats[0].at<double>(0, 0); 

    RSL_R_Mats = loadCamFiles("cam-params/rsleft_right_extrinsics.yml", "cam-params/rsleft_right_intrinsics.yml");
    right_color_cam_matrix = RSL_R_Mats[1];
    right_color_distort = RSL_R_Mats[3];
    Mat T = RSL_R_Mats[8];
    Mat R = RSL_R_Mats[9];
    
    Rt_right.at<double>(0, 3) = T.at<double>(0)/Tdiv;
    Rt_right.at<double>(1, 3) = T.at<double>(1)/Tdiv;
    Rt_right.at<double>(2, 3) = T.at<double>(2)/Tdiv;
    Rt_right.at<double>(0, 0) = R.at<double>(0, 0) * Rtmul; Rt_right.at<double>(0, 1) = R.at<double>(0, 1) * Rtmul; Rt_right.at<double>(0, 2) = R.at<double>(0, 2) * Rtmul;
    Rt_right.at<double>(1, 0) = R.at<double>(1, 0) * Rtmul; Rt_right.at<double>(1, 1) = R.at<double>(1, 1) * Rtmul; Rt_right.at<double>(1, 2) = R.at<double>(1, 2) * Rtmul;
    Rt_right.at<double>(2, 0) = R.at<double>(2, 0) * Rtmul; Rt_right.at<double>(2, 1) = R.at<double>(2, 1) * Rtmul; Rt_right.at<double>(2, 2) = R.at<double>(2, 2) * Rtmul;
    Rt_right.at<double>(3, 0) = 0.0f; Rt_right.at<double>(3, 1) = 0.0f; Rt_right.at<double>(3, 2) = 0.0f; Rt_right.at<double>(3, 3) = 1.0f;

    RSL_L_Mats = loadCamFiles("cam-params/rsleft_left_extrinsics.yml", "cam-params/rsleft_left_intrinsics.yml");
    left_color_cam_matrix = RSL_L_Mats[1];
    left_color_distort = RSL_L_Mats[3];
    T = RSL_L_Mats[8];
    R = RSL_L_Mats[9];
    
    Rt_left.at<double>(0, 3) = T.at<double>(0)/Tdiv;
    Rt_left.at<double>(1, 3) = T.at<double>(1)/Tdiv;
    Rt_left.at<double>(2, 3) = T.at<double>(2)/Tdiv;

    Rt_left.at<double>(0, 0) = R.at<double>(0, 0); Rt_left.at<double>(0, 1) = R.at<double>(0, 1); Rt_left.at<double>(0, 2) = R.at<double>(0, 2);
    Rt_left.at<double>(1, 0) = R.at<double>(1, 0); Rt_left.at<double>(1, 1) = R.at<double>(1, 1); Rt_left.at<double>(1, 2) = R.at<double>(1, 2);
    Rt_left.at<double>(2, 0) = R.at<double>(2, 0); Rt_left.at<double>(2, 1) = R.at<double>(2, 1); Rt_left.at<double>(2, 2) = R.at<double>(2, 2);
    Rt_left.at<double>(3, 0) = 0.0f; Rt_left.at<double>(3, 1) = 0.0f; Rt_left.at<double>(3, 2) = 0.0f; Rt_left.at<double>(3, 3) = 1.0f;
}

void initRealSense() {
    // Use a configuration object to request only depth from the pipeline
    cfg.enable_stream(RS2_STREAM_DEPTH, resX, resY, RS2_FORMAT_Z16, 30);
    // Start streaming with the above configuration
    auto profile = pipe.start(cfg);

    auto sensor = profile.get_device().first<rs2::depth_sensor>();
    auto depth_scale = sensor.get_depth_scale();

    float clipping_distance_in_meters = 1.0; //#1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale;

    auto i_profile = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    rs2_intrinsics intr = i_profile.as<rs2::video_stream_profile>().get_intrinsics();

    // ############  RS LOCAL CAMERA MATRIX 
    depth_cam_matrix.at<double>(0, 0) = intr.fx;  //#fx
    depth_cam_matrix.at<double>(0, 2) = intr.ppx; //#cx
    depth_cam_matrix.at<double>(1, 1) = intr.fy; //#fy
    depth_cam_matrix.at<double>(1, 2) = intr.ppy; // #cy
    depth_cam_matrix.at<double>(2, 2) = 1.0;

    //  ############ REAL SENSE INIT VARIABLES ######################       
    
}

int main(int, char**)
{
    // Mats for the cameras frames
    Mat f_left;  //left camera
    Mat f_right; //right camera
    Mat f_center; //center camera
    Mat rs_depth; //rs depth information

    initVariables();  // INIT GLOABL MATRICES AND VARIABLES 

    initRealSense();  // INIT REALSENSE CONTROL OBJECTS AND VARIABLES

    // ##################    INIT CAMERA THREADS   ####################
    cameraThread* lCam = new cameraThread(resX, resY, 3);  //left camera
    cameraThread* cCam = new cameraThread(resX, resY, 2);  //right camera
    cameraThread* rCam = new cameraThread(resX, resY, 0);  //center camera

    thread lCamTh(&cameraThread::camLoop, lCam, 0);
    thread cCamTh(&cameraThread::camLoop, cCam, 0);
    thread rCamTh(&cameraThread::camLoop, rCam, 0);
    // ##################    INIT CAMERA THREADS   ####################

    // ###################  INIT GPU THREADS #########################
    // M1, M2, D1, D2, R1, R2, P1, P2, T, R
    // 0=mapx1_left,1=mapy1_left,2=mapx2_left,3=mapy2_left, 4=cutMatrixLeft
    vector<Mat> mleft;
    mleft.push_back(mapx1_left); mleft.push_back(mapy1_left); mleft.push_back(mapx2_left); mleft.push_back(mapy2_left); //mleft.push_back(cutMatrixLeft);
    vector<Mat> mright;
    mright.push_back(mapx1_right); mright.push_back(mapy1_right); mright.push_back(mapx2_right); mright.push_back(mapy2_right); //mright.push_back(cutMatrixRight);
    cuStream* gpu1 = new cuStream(resX, resY, mleft, mright, base_rgb_left, base_rgb_right, clipping_distance, 'h', true, 'l');
    thread gpu1Th(&cuStream::streamLoop, gpu1, 0);

    cuStream* gpu2 = new cuStream(resX, resY, mleft, mright, base_rgb_left, base_rgb_right, clipping_distance, 'h', true, 'r');
    thread gpu2Th(&cuStream::streamLoop, gpu2, 0);



    // ###################  INIT GPU THREADS #########################
    try {
        for (;;)  // ################ THE MAIN PROGRAM LOOP ########################
        {
            f_left   = lCam->getFrame();
            f_center = cCam->getFrame();
            f_right  = rCam->getFrame();

            rs2::frameset frameset = pipe.wait_for_frames();
            rs2::frame depth = frameset.get_depth_frame();

            if (f_left.empty() || f_center.empty() || f_right.empty() || !depth ) {  // CHECK IF WE HAVE FRAMES
                std::cout << "ERROR: Blank frames grabbed... \n" << std::endl;
                continue;
            } 
            Mat depth_image(Size(resX, resY), CV_16UC1, (void*)depth.get_data(), Mat::AUTO_STEP);  // GET THE DEPTH DATA IN Z16 FORMAT
            
            // #### ADD ALPHA CHANNEL TO THE CAMERA IMAGES
            cv::cvtColor(f_left, f_left, COLOR_RGB2RGBA);
            cv::cvtColor(f_center, f_center, COLOR_RGB2RGBA);
            cv::cvtColor(f_right, f_right, COLOR_RGB2RGBA);

            // #####   REGISTER THE DEPTH WITH THE LEFT AND RIGHT CAMERAS  POV #############
            cv::rgbd::registerDepth(depth_cam_matrix, left_color_cam_matrix, Mat(), Rt_left, depth_image, cv::Size(resX, resY), depth_image_left, true);
            cv::rgbd::registerDepth(depth_cam_matrix, right_color_cam_matrix, Mat(), Rt_right, depth_image, cv::Size(resX, resY), depth_image_right, true);
            // #####   REGISTER THE DEPTH WITH THE LEFT AND RIGHT CAMERAS  POV #############

            //std::cout << f_center.size() << std::endl;

            gpu1->updateImages(depth_image_left, f_center, f_left, 'l');
            gpu2->updateImages(depth_image_right, f_center, f_right, 'r');
            
            if (gpu1->isDone() || gpu2->isDone()) {
                std::cout << "Killing all the control threads, please wait... " << std::endl;
                lCam->stop(); cCam->stop(); rCam->stop(); gpu1->stop(); gpu2->stop();
                lCamTh.join();  cCamTh.join(); rCamTh.join(); gpu1Th.join(); gpu2Th.join();
                break;
            }
        }
    }
    catch (std::exception& e) {
        std::cout << "CPU exception: " << e.what() << std::endl;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
