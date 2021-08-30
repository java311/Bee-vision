#include <vector>
#include <iostream>       // std::cout
#include <queue>
#include <stdio.h>

using namespace cv;
using namespace std;

class cuStream {
private:
	cv::cuda::Stream stream;
	int resX; 
	int resY; 
	vector<Mat> matrices_left;
	vector<Mat> matrices_right; 
	double base_rgb_uv_left; 
	double base_rgb_uv_right; 
	double clip_dist; 
	char orientation;
	bool add;
	char side;
	bool done;

	//Update Mats for the depth, sensor and rgb images
	Mat color_image;  //rgb image
	Mat depth_image;  //depth data (float)
	Mat uv_image;     //sensor image

	bool stopFlag;
	queue<Mat> depth_image_queue;
	queue<Mat> uv_image_queue;
	queue<Mat> color_image_queue;


	// ############  GPU OBJECTS AND MATRICES ###################
	cv::cuda::GpuMat cuMask;
	cv::cuda::GpuMat cuUv;
	cv::cuda::GpuMat cuColor_left;
	cv::cuda::GpuMat cuColor_right;

	// GPU matrices for MASKING 
	cv::cuda::GpuMat cuMaskDist; 
	cv::cuda::GpuMat cuMaskZeros;
	cv::cuda::GpuMat cuClip1; 
	cv::cuda::GpuMat cuClip2;

	// MASKING FILTERS pointers !!!
	cv::Ptr<cv::cuda::Filter>cuErosionFilter;  //opencv pointers used here
	cv::Ptr<cv::cuda::Filter>cuDilateFilter;
	cv::Ptr<cv::cuda::Filter>cuGaussFilter;

	// DISPARITY GPU MATRICES
	cv::cuda::GpuMat ones_f;
	cv::cuda::GpuMat cuDisp_left;  // disparity warp left
	cv::cuda::GpuMat cuDisp_left_tmp;  // disparity warp left
	cv::cuda::GpuMat cuMapx1_left;
	cv::cuda::GpuMat cuMapy1_left;
	cv::cuda::GpuMat cuMapx2_left;
	cv::cuda::GpuMat cuMapy2_left;
	cv::cuda::GpuMat bases_left;
	cv::cuda::GpuMat cuDisp_right;  // disparity warp right
	cv::cuda::GpuMat cuDisp_right_tmp; 
	cv::cuda::GpuMat cuMapx1_right;
	cv::cuda::GpuMat cuMapy1_right;
	cv::cuda::GpuMat cuMapx2_right;
	cv::cuda::GpuMat cuMapy2_right;
	cv::cuda::GpuMat bases_right;

	cv::cuda::GpuMat cuGrid_x;
	cv::cuda::GpuMat cuGrid_y;
	cv::cuda::GpuMat cuRecColor_left;
	cv::cuda::GpuMat cuRecColor_right;
	cv::cuda::GpuMat cuRecUv_left; 
	cv::cuda::GpuMat cuRecUv_right;
	cv::cuda::GpuMat cuRecMask_left;
	cv::cuda::GpuMat cuRecMask_right;
	cv::cuda::GpuMat cuResult_left;
	cv::cuda::GpuMat cuResult_right;
	cv::cuda::GpuMat cuResult_left_r;
	cv::cuda::GpuMat cuResult_right_r;
	cv::cuda::GpuMat cuFinal_left;
	cv::cuda::GpuMat cuFinal_right;




public:
	// Construction
	cuStream(int resX, int resY, vector<Mat> matrices_left, vector<Mat> matrices_right, double base_rgb_uv_left, double base_rgb_uv_right, double clip_dist, char orientation, bool add, char side)
	{
		this->done = false;
		this->resX = resX;
		this->resY = resY;
		this->matrices_left = matrices_left;
		this->matrices_right = matrices_right;
		this->base_rgb_uv_left = base_rgb_uv_left;
		this->base_rgb_uv_right = base_rgb_uv_right;
		this->clip_dist = clip_dist;
		this->orientation = orientation;
		this->add = add;
		this->side = side;

		this->stopFlag = false;

		// ###### MASKING FILTERS (for mask creation)
		int eSize = 5;
		Mat e_element = cv::getStructuringElement(MORPH_RECT, Size(2 * eSize + 1, 2 * eSize + 1), Point(eSize, eSize));
		this->cuErosionFilter = cv::cuda::createMorphologyFilter(MORPH_ERODE, CV_8UC1, e_element);
		int dSize = 15;
		Mat d_element = cv::getStructuringElement(MORPH_RECT, Size(2 * dSize + 1, 2 * dSize + 1), Point(dSize, dSize));
		this->cuDilateFilter = cv::cuda::createMorphologyFilter(MORPH_DILATE, CV_8UC1, d_element);
		int gSize = 15;
		this->cuGaussFilter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(gSize, gSize), 0, 0);

		// ##### MASKING VARIABLES  
		cuMaskDist.upload(cv::Mat(cv::Size(resX, resY), CV_16UC1, cv::Scalar(int(clip_dist))), this->stream);  //init with clipping distance
		cuMaskZeros.upload(cv::Mat(cv::Size(resX, resY), CV_16UC1, cv::Scalar(int(0))), this->stream);         //init with zeros

		// ##### MATRICES FOR WARPING 
		//0=mapx1_left,1=mapy1_left,2=mapx2_left,3=mapy2_left, 4=cutMatrixLeft
		this->cuMapx1_left.upload(matrices_left[0], this->stream);   // STATIC
		this->cuMapy1_left.upload(matrices_left[1], this->stream);   // STATIC
		this->cuMapx2_left.upload(matrices_left[2], this->stream);   // STATIC
		this->cuMapy2_left.upload(matrices_left[3], this->stream);   // STATIC

		this->cuMapx1_right.upload(matrices_right[0], this->stream);   // STATIC
		this->cuMapy1_right.upload(matrices_right[1], this->stream);   // STATIC
		this->cuMapx2_right.upload(matrices_right[2], this->stream);   // STATIC
		this->cuMapy2_right.upload(matrices_right[3], this->stream);   // STATIC

		this->bases_left.upload(cv::Mat(cv::Size(resX, resY), CV_32F, cv::Scalar(this->base_rgb_uv_left)), this->stream);   //STATIC 
		this->bases_right.upload(cv::Mat(cv::Size(resX, resY), CV_32F, cv::Scalar(this->base_rgb_uv_right)), this->stream);   //STATIC 

		this->ones_f.upload(cv::Mat(cv::Size(resX, resY), CV_32F, cv::Scalar(1.0)), this->stream);//gpu mat with ones (for disparity computation)

		Mat grid_x = Mat::zeros(cv::Size(resX, resY), CV_32F);
		Mat grid_y = Mat::zeros(cv::Size(resX, resY), CV_32F);
		this->fillIndices(grid_x, 'x');
		this->fillIndices(grid_y, 'y');
		//std::cout << grid_x << std::endl;
		//std::cout << grid_y << std::endl;
		cv::FileStorage file("debug.ext", cv::FileStorage::WRITE);
		// Write to file!
		file << "grid_x" << grid_x;
		file << "grid_y" << grid_y;
		file.release();

		this->cuGrid_x.upload(grid_x, this->stream);   //STATIC
		this->cuGrid_y.upload(grid_y, this->stream);   //STATIC

		this->cuRecColor_left.upload(Mat::zeros(cv::Size(resX, resY), CV_32FC4), this->stream);
		this->cuRecColor_right.upload(Mat::zeros(cv::Size(resX, resY), CV_32FC4), this->stream);
		this->cuRecUv_left.upload(Mat::zeros(cv::Size(resX, resY), CV_8UC4), this->stream);
		this->cuRecUv_right.upload(Mat::zeros(cv::Size(resX, resY), CV_8UC4), this->stream);
		this->cuRecMask_left.upload(Mat::zeros(cv::Size(resX, resY), CV_8U), this->stream);
		this->cuRecMask_right.upload(Mat::zeros(cv::Size(resX, resY), CV_8U), this->stream);

		this->cuResult_left.upload(Mat::zeros(cv::Size(resX, resY), CV_8UC4), this->stream);
		this->cuResult_right.upload(Mat::zeros(cv::Size(resX, resY), CV_8UC4), this->stream);
		this->cuResult_left_r.upload(Mat::zeros(cv::Size(resX, resY), CV_8UC4), this->stream);
		this->cuResult_right_r.upload(Mat::zeros(cv::Size(resX, resY), CV_8UC4), this->stream);

		this->cuFinal_left.upload(Mat::zeros(cv::Size(resX, resY), CV_8UC4), this->stream);
		this->cuFinal_right.upload(Mat::zeros(cv::Size(resX, resY), CV_8UC4), this->stream);

		/*
		self.cuRecColor_left.upload(np.full((resY, resX), 0, dtype="float32"), stream=self.stream)
        self.cuRecColor_right.upload(np.full((resY, resX), 0, dtype="float32"), stream=self.stream)
        # self.cuRecUv = cv2.cuda_GpuMat()
        self.cuRecUv_left.upload(np.full((resY, resX, 4), 0, dtype="float32"), stream=self.stream)
        self.cuRecUv_right.upload(np.full((resY, resX, 4), 0, dtype="float32"), stream=self.stream)
        # self.cuRecMask = cv2.cuda_GpuMat()  
        self.cuRecMask_left.upload(np.full((resY, resX), 0, dtype="float32"), stream=self.stream)
        self.cuRecMask_right.upload(np.full((resY, resX), 0, dtype="float32"), stream=self.stream)
        # self.cuResult = cv2.cuda_GpuMat()  
        self.cuResult_left.upload(np.full((resY, resX, 4), 0, dtype="float32"), stream=self.stream)
        self.cuResult_right.upload(np.full((resY, resX, 4), 0, dtype="float32"), stream=self.stream)
        # self.cuFinal = cv2.cuda_GpuMat()  
        self.cuFinal_left.upload( np.full((resY, resX, 4), 0, dtype="float32"), stream=self.stream)
        self.cuFinal_right.upload( np.full((resY, resX, 4), 0, dtype="float32"), stream=self.stream)
		*/
	}

	void fillIndices(Mat& matrix, char orientation)
	{
		int rows = matrix.size().height;
		int cols = matrix.size().width;

		for (int col = 0; col < cols; col=col+1) {
			for (int row = 0; row < rows; row=row+1) {
				if (orientation == 'y') {
					matrix.at<float>(row, col) = row;
				}
				else {
					matrix.at<float>(row, col) = col;
				}
					
			}
		}
	}

	bool isDone() {
		return this->done;
	}

	void stop() {
		stopFlag = true;
		cout << "Stop GPU ["<< this->side << "] thread..." << std::endl;
	}

	void updateImages(Mat depth_image, Mat uv_image, Mat color_image, char side) {
		this->depth_image_queue.push(depth_image);
		this->uv_image_queue.push(uv_image);
		this->color_image_queue.push(color_image);
		this->side = side;
	}	

	void streamLoop(int x)
	{
		try {
			while (!stopFlag) {
				if (depth_image_queue.empty() || uv_image_queue.empty() || color_image_queue.empty()) {
					continue;
				}

				// ##################### Remove background  GPU ################################ 
				this->depth_image = depth_image_queue.front();
				depth_image_queue.pop();

				this->cuMask.upload(this->depth_image, this->stream);
				cv::cuda::compare(this->cuMask, this->cuMaskDist, this->cuClip1, CMP_LE, this->stream);
				cv::cuda::compare(this->cuMask, this->cuMaskZeros, this->cuClip2, CMP_GT, this->stream);
				cv::cuda::bitwise_and(this->cuClip1, this->cuClip2, this->cuMask, noArray(), this->stream);

				this->cuErosionFilter->apply(this->cuMask, this->cuMask, this->stream);
				this->cuDilateFilter->apply(this->cuMask, this->cuMask, this->stream);
				this->cuGaussFilter->apply(this->cuMask, this->cuMask, this->stream);

				//############### GET DISPARITY MATRIX for UV CAMERAS ########################################### 
				
				
				// Remap of the depth map
				if (this->side == 'l') {  //LEFT DISAPRITY CALC
					Mat depth_image_left_f;
					this->depth_image.convertTo(depth_image_left_f, CV_32F); //cast to floats
					this->cuDisp_left_tmp.upload(depth_image_left_f, this->stream);
					cv::cuda::remap(this->cuDisp_left_tmp, this->cuDisp_left, this->cuMapx1_left, this->cuMapy1_left, INTER_LINEAR, 0, Scalar(), this->stream);
					cv::cuda::divide(this->ones_f, this->cuDisp_left, this->cuDisp_left, 1.0, -1, this->stream);
					cv::cuda::multiply(this->cuDisp_left, this->bases_left, this->cuDisp_left, 1.0, -1, this->stream);
				}else {                   //RIGHT DISPARITY CALC
					Mat depth_image_right_f;
					this->depth_image.convertTo(depth_image_right_f, CV_32F); //cast to floats
					this->cuDisp_right_tmp.upload(depth_image_right_f, this->stream);
					cv::cuda::remap(this->cuDisp_right_tmp, this->cuDisp_right, this->cuMapx1_right, this->cuMapy1_right, INTER_LINEAR, 0, Scalar(), this->stream);
					cv::cuda::divide(this->ones_f, this->cuDisp_right, this->cuDisp_right, 1.0, -1, this->stream);
					cv::cuda::multiply(this->cuDisp_right, this->bases_right, this->cuDisp_right, 1.0, -1, this->stream);
				}
				
				// Transform the depth map to disparity map aligned to the external camera
				if (this->orientation == 'h') {
					if (this->add) {
						if (this->side == 'l') {
							cv::cuda::add(this->cuGrid_x, this->cuDisp_left, this->cuDisp_left_tmp, noArray(), -1, this->stream);
						}
						else {
							cv::cuda::add(this->cuGrid_x, this->cuDisp_right, this->cuDisp_right_tmp, noArray(), -1, this->stream);
						}
					}
					else {
						if (this->side == 'l') {
							cv::cuda::subtract(this->cuGrid_x, this->cuDisp_left, this->cuDisp_left_tmp, noArray(), -1, this->stream);
						}
						else {
							cv::cuda::subtract(this->cuGrid_x, this->cuDisp_right, this->cuDisp_right_tmp, noArray(), -1, this->stream);
						}
					}
				}else {
					if (this->add) {
						if (this->side == 'l') {
							cv::cuda::add(this->cuGrid_y, this->cuDisp_left, this->cuDisp_left_tmp, noArray(), -1, this->stream);
						}
						else {
							cv::cuda::add(this->cuGrid_y, this->cuDisp_right, this->cuDisp_right_tmp, noArray(), -1, this->stream);
						}
					}
					else {
						if (this->side == 'l') {
							cv::cuda::subtract(this->cuGrid_y, this->cuDisp_left, this->cuDisp_left_tmp, noArray(), -1, this->stream);
						}
						else {
							cv::cuda::subtract(this->cuGrid_y, this->cuDisp_right, this->cuDisp_right_tmp, noArray(), -1, this->stream);
						}
					}
				}
				
				// ############## Rectify the UV, RGB and Mask images ##################################
				this->uv_image = uv_image_queue.front();
				this->cuUv.upload(this->uv_image, this->stream);
				uv_image_queue.pop();

				if (this->side == 'l') {
					this->color_image = this->color_image_queue.front();
					this->cuColor_left.upload(this->color_image, this->stream);
					color_image_queue.pop();
					cv::cuda::remap(this->cuColor_left, this->cuRecColor_left, this->cuMapx1_left, this->cuMapy1_left, INTER_LINEAR, 0, Scalar(), this->stream);  //rgb
					cv::cuda::remap(this->cuUv, this->cuRecUv_left, this->cuMapx2_left, this->cuMapy2_left, INTER_LINEAR, 0, Scalar(), this->stream);  // uv
					cv::cuda::remap(this->cuMask, this->cuRecMask_left, this->cuMapx1_left, this->cuMapy1_left, INTER_LINEAR, 0, Scalar(), this->stream);
				}
				else {
					this->color_image = this->color_image_queue.front();
					this->cuColor_right.upload(this->color_image, this->stream);
					color_image_queue.pop();
					cv::cuda::remap(this->cuColor_right, this->cuRecColor_right, this->cuMapx1_right, this->cuMapy1_right, INTER_LINEAR, 0, Scalar(), this->stream);  //rgb
					cv::cuda::remap(this->cuUv, this->cuRecUv_right, this->cuMapx2_right, this->cuMapy2_right, INTER_LINEAR, 0, Scalar(), this->stream);  // uv
					cv::cuda::remap(this->cuMask, this->cuRecMask_right, this->cuMapx1_right, this->cuMapy1_right, INTER_LINEAR, 0, Scalar(), this->stream);
				}

				
				// # Make the remaping of the UV image over the RGB image using the computed disparity Map
				if (this->orientation == 'h') {
					if (this->side == 'l') {
						cv::cuda::remap(this->cuRecUv_left, this->cuResult_left, this->cuDisp_left_tmp, this->cuGrid_y, INTER_LINEAR, 0, Scalar(), this->stream);  //returns a uint8
					}
					else {
						cv::cuda::remap(this->cuRecUv_right, this->cuResult_right, this->cuDisp_right_tmp, this->cuGrid_y, INTER_LINEAR, 0, Scalar(), this->stream);  //returns a uint8
					}
				}
				else {
					if (this->side == 'l') {
						cv::cuda::remap(this->cuRecUv_left, this->cuResult_left, this->cuGrid_x, this->cuDisp_left_tmp, INTER_LINEAR, 0, Scalar(), this->stream);  //returns a uint8
					}
					else {
						cv::cuda::remap(this->cuRecUv_right, this->cuResult_right, this->cuGrid_x, this->cuDisp_right_tmp, INTER_LINEAR, 0, Scalar(), this->stream);  //returns a uint8
					}
				}
				
				// ###############   Mix the MASK and the rectified RGB and UV images ############
				if (this->side == 'l') {
					Mat resu, recMask;
					this->cuRecMask_left.download(recMask, this->stream);
					this->cuResult_left.download(resu, this->stream);
					cv::convertScaleAbs(recMask, recMask, 255, 0);   //recMask.convertTo(recMask, CV_32F);
					cv::insertChannel(recMask, resu, 3);
					this->cuResult_left.upload(resu, this->stream);

					cv::cuda::alphaComp(this->cuResult_left, this->cuRecColor_left, this->cuFinal_left, cv::cuda::ALPHA_OVER, this->stream);

					// ############### CUT THE WARP DISTORTION IN THE FINAL IMAGE ###################################
					// self.cuFinal_left = cv2.cuda.warpPerspective(self.cuFinal_left, self.matrices_left.cutMatrixUv, (self.resX, self.resY), stream = self.stream)
					// ############### CUT THE WARP DISTORTION IN THE FINAL IMAGE ###################################
				}
				else {
					Mat resu, recMask;
					this->cuRecMask_right.download(recMask, this->stream);
					this->cuResult_right.download(resu, this->stream);
					cv::convertScaleAbs(recMask, recMask, 255, 0);   //recMask.convertTo(recMask, CV_32F);
					cv::insertChannel(recMask, resu, 3);
					this->cuResult_right.upload(resu, this->stream);

					cv::cuda::alphaComp(this->cuResult_right, this->cuRecColor_right, this->cuFinal_right, cv::cuda::ALPHA_OVER, this->stream);

					// ############### CUT THE WARP DISTORTION IN THE FINAL IMAGE ###################################
					// self.cuFinal_left = cv2.cuda.warpPerspective(self.cuFinal_left, self.matrices_left.cutMatrixUv, (self.resX, self.resY), stream = self.stream)
					// ############### CUT THE WARP DISTORTION IN THE FINAL IMAGE ###################################
				}
				
				//Mat pinchi = cv::Mat::zeros(cv::Size(resX, resY), CV_8UC4);
				//Mat  p1, p2, p3, p4, p5;
				//this->cuRecColor_left.download(p1, this->stream);
				//this->cuRecUv_left.download(p2, this->stream);
				//this->cuRecMask_left.download(p3, this->stream);
				//this->cuFinal_left.download(p4, this->stream);
				//this->cuResult_left.download(p5, this->stream);
				/*this->cuMask.download(p1);*/
				//imshow("cuRecColor", p1);
				//imshow("cuRecUv", p2);
				//imshow("cuRecMask", p3);
				//imshow("Final", p4);
				//imshow("cuResult_left", p5);
				Mat final;
				if (this->side == 'l') {
					this->cuFinal_left.download(final, this->stream);
					cv::resize(final, final, Size(int(resX / 2), int(resY / 2)));
				}
				else {
					this->cuFinal_right.download(final, this->stream);
					cv::resize(final, final, Size(int(resX / 2), int(resY / 2)));
				}
				string title = "Final_";
				title += this->side;
				imshow(title, final);
				if (waitKey(5) >= 0) {
					break;
				}
			}

		}
		catch (std::exception& e) {
			std::cout << "GPU exception: " << e.what() << " " << std::endl;
			this->done = true;
		}
		this->done = true;
	}
};

#pragma once
