import threading
import numpy as np
import cv2
import time 

class Matrices():
    def __init__(self, mapx1_uv, mapy1_uv, mapx2_uv, mapy2_uv,cutMatrixUv):
        self.mapx1_uv = mapx1_uv
        self.mapy1_uv = mapy1_uv
        self.mapx2_uv = mapx2_uv
        self.mapy2_uv = mapy2_uv
        self.cutMatrixUv = cutMatrixUv

class cuStream(threading.Thread):
    #CLASS CONTROL VARIABLEs
    stream = None
    resX = 1280
    resY = 720
    event = None
    matrices = None
    orientation = 'h' 
    add = True   

    ##IMAGES
    color_image = None
    uv_image = None
    depth_image = None

    #GPU VARIABLES
    ####IMAGES
    cuDepth = cv2.cuda_GpuMat()
    cuUv = cv2.cuda_GpuMat() 
    cuColor = cv2.cuda_GpuMat()
    cuDisp = cv2.cuda_GpuMat()
    ####MAPPING
    cuMapx1_uv = cv2.cuda_GpuMat()
    cuMapy1_uv = cv2.cuda_GpuMat()
    cuMapx2_uv = cv2.cuda_GpuMat()
    cuMapy2_uv = cv2.cuda_GpuMat()
    ones_f = cv2.cuda_GpuMat()
    bases = cv2.cuda_GpuMat()
    cuGrid_x = cv2.cuda_GpuMat()
    cuGrid_y = cv2.cuda_GpuMat()
    ####RECTIFY AND REMAP BLOCK
    cuRecColor = cv2.cuda_GpuMat()
    cuRecUv = cv2.cuda_GpuMat()
    cuRecMask = cv2.cuda_GpuMat()
    cuResult = cv2.cuda_GpuMat()
    cuFinal = cv2.cuda_GpuMat()
    ones = cv2.cuda_GpuMat()
    # MASKING
    cuMask = cv2.cuda_GpuMat()
    cuClip1 = cv2.cuda_GpuMat()
    cuClip2 = cv2.cuda_GpuMat()
    cuMaskDist = cv2.cuda_GpuMat()
    cuMaskZeros = cv2.cuda_GpuMat()
    cuErosionFilter = None
    cuDilateFilter = None
    cuGaussFilter = None

    def __init__(self, stream, resX, resY, matrices, base_rgb_uv, clip_dist, orientation, add, event):
        # GPU MATRICES INIT
        self.stream = stream
        self.resX = resX
        self.resY = resY
        self.matrices = matrices
        self.event = event
        self.orientation = orientation
        self.add = add

        # self.cuDepth = cv2.cuda_GpuMat() 
        # self.secuUv = cv2.cuda_GpuMat() 
        # self.cuColor = cv2.cuda_GpuMat()
        # self.cuDisp = cv2.cuda_GpuMat(); 
        self.cuDisp.upload(np.full((resY, resX), 0, dtype="float32"))  #this static init may cause a bug
        # self.cuMapx1_uv = cv2.cuda_GpuMat(); 
        self.cuMapx1_uv.upload(matrices.mapx1_uv)  #these are static
        # self.cuMapy1_uv = cv2.cuda_GpuMat(); 
        self.cuMapy1_uv.upload(matrices.mapy1_uv)  #these are static
        # self.cuMapx2_uv = cv2.cuda_GpuMat();  
        self.cuMapx2_uv.upload(matrices.mapx2_uv)  #these are static
        # self.cuMapy2_uv = cv2.cuda_GpuMat();  
        self.cuMapy2_uv.upload(matrices.mapy2_uv)  #these are static

        # self.ones_f = cv2.cuda_GpuMat();  
        self.ones_f.upload(np.full((resY, resX), 1.0, dtype="float32"))          #static
        # self.bases = cv2.cuda_GpuMat();  
        self.bases.upload(np.full((resY, resX), base_rgb_uv, dtype="float32"))    #static

        grid = np.indices((resY, resX))                                                   #static
        # self.cuGrid_x = cv2.cuda_GpuMat()  
        self.cuGrid_x.upload(grid[1].astype(np.float32))        #static
        # self.cuGrid_y = cv2.cuda_GpuMat()  
        self.cuGrid_y.upload(grid[0].astype(np.float32))        #static


        # Rectify and remap block 
        # self.cuRecColor = cv2.cuda_GpuMat(); 
        self.cuRecColor.upload(np.full((resY, resX), 0, dtype="float32"))
        # self.cuRecUv = cv2.cuda_GpuMat()
        self.cuRecUv.upload(np.full((resY, resX, 4), 0, dtype="float32"))
        # self.cuRecMask = cv2.cuda_GpuMat()  
        self.cuRecMask.upload(np.full((resY, resX), 0, dtype="float32"))
        # self.cuResult = cv2.cuda_GpuMat()  
        self.cuResult.upload(np.full((resY, resX, 4), 0, dtype="float32"))
        # self.cuFinal = cv2.cuda_GpuMat()  
        self.cuFinal.upload( np.full((resY, resX, 4), 0, dtype="float32"))
        # self.ones = cv2.cuda_GpuMat()  
        self.ones.upload(np.full((resY, resX, 3), 1, dtype="float32") )

        #######  MASKING
        # self.cuMask = cv2.cuda_GpuMat()
        self.cuMask.upload(np.full((resY, resX), 0, dtype="uint16") )
        # self.cuClip1 = cv2.cuda_GpuMat()  
        self.cuClip1.upload(np.full((resY, resX), 0, dtype="uint8") )
        # self.cuClip2 = cv2.cuda_GpuMat()  
        self.cuClip2.upload(np.full((resY, resX), 0, dtype="uint8") )
        # self.cuMaskDist = cv2.cuda_GpuMat() 
        self.cuMaskDist.upload(np.full((resY, resX), int(clip_dist), dtype="uint16") )
        # self.cuMaskZeros = cv2.cuda_GpuMat() 
        self.cuMaskZeros.upload(np.full((resY, resX), 0, dtype="uint16") )

        ###### MASKING FILTERS (for mask creation)
        eSize = 5
        e_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*eSize + 1, 2*eSize+1), (eSize, eSize))
        self.cuErosionFilter = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE, cv2.CV_8UC1, e_element)
        dSize = 15
        d_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dSize + 1, 2*dSize+1), (dSize, dSize))
        self.cuDilateFilter = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, d_element)
        gSize = 15
        self.cuGaussFilter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (gSize, gSize), 0, 0)

        ############# THREAD CONTROL
        threading.Thread.__init__(self, name='gpu1')

    def updateImages(self, color_image, uv_image, depth_image):
        self.color_image = color_image
        self.uv_image = uv_image
        self.depth_image = depth_image

    def run(self):
        # try:
        while True:
            event_is_set = self.event.wait()
            if event_is_set:
                start = time.time_ns()
                # ##################### Remove background  GPU ################################  17ms   
                # mask = np.where((depth_image > clipping_distance) | (depth_image <= 0), 0.0, alphaValue)   # orignal formula	
                self.cuMask.upload(np.array(self.depth_image))
                self.cuClip1 = cv2.cuda.compare(self.cuMask, self.cuMaskDist, cv2.CMP_LE, stream=self.stream)   #depth < clipping_distance (inverted)
                self.cuClip2 = cv2.cuda.compare(self.cuMask, self.cuMaskZeros, cv2.CMP_GT, stream=self.stream)  #depth >= 0   (inverted)
                self.cuMask = cv2.cuda.bitwise_and(self.cuClip1, self.cuClip2, stream=self.stream)   #bitwise OR
                
                self.cuMask = self.cuErosionFilter.apply(self.cuMask, stream=self.stream)
                self.cuMask = self.cuDilateFilter.apply(self.cuMask, stream=self.stream)
                self.cuMask = self.cuGaussFilter.apply(self.cuMask, stream=self.stream)
                # ##################### Remove background  GPU ################################ 13ms

                ############### GET DISPARITY GPU for UV CAMERA ########################################### This step takes 5ms 
        
                # Make all the uploads necessary (depth, mapx) 
                self.cuDepth.upload(self.depth_image.astype(np.float32))  #CHECK THIS, THE CAST WAS ORIGINALLY AFTER THE REMAP

                #Remap of the depth map
                self.cuDisp = cv2.cuda.remap(self.cuDepth, self.cuMapx1_uv, self.cuMapy1_uv, cv2.INTER_LINEAR, stream=self.stream)
                self.cuDisp = cv2.cuda.divide(self.ones_f, self.cuDisp, stream=self.stream)
                self.cuDisp = cv2.cuda.multiply(self.cuDisp, self.bases, stream=self.stream)

                #Transform the depth map to disparity map aligned to the external camera 
                if self.orientation == 'h':
                    if self.add:
                        self.cuDisp = cv2.cuda.add(self.cuGrid_x, self.cuDisp, stream=self.stream)
                    else:
                        self.cuDisp = cv2.cuda.subtract(self.cuGrid_x, self.cuDisp, stream=self.stream)
                elif self.orientation == 'v':
                    if self.add:
                        self.cuDisp = cv2.cuda.add(self.cuGrid_y, self.cuDisp, stream=self.stream)
                    else:
                        self.cuDisp = cv2.cuda.subtract(self.cuGrid_y, self.cuDisp, stream=self.stream)
                
                ############### GET DISPARITY GPU for UV CAMERA ###########################################

                ############### Rectify the UV, RGB and Mask images ########################################### Both rectify and remap sections take around 5ms
        
                self.cuUv.upload(self.uv_image)
                self.cuColor.upload(self.color_image)
                
                self.cuRecColor = cv2.cuda.remap(self.cuColor, self.cuMapx1_uv, self.cuMapy1_uv, cv2.INTER_LINEAR, stream=self.stream)  #rgb
                self.cuRecUv = cv2.cuda.remap(self.cuUv, self.cuMapx2_uv, self.cuMapy2_uv, cv2.INTER_LINEAR, stream=self.stream) #uv
                self.cuRecMask = cv2.cuda.remap(self.cuMask, self.cuMapx1_uv, self.cuMapy1_uv, cv2.INTER_LINEAR, stream=self.stream) #mask aligned to rgb
                
                ###############   Make the remaping of the UV image over the RGB image using the computed disparity Map
                if self.orientation == 'h':
                    self.cuResult = cv2.cuda.remap(self.cuRecUv, self.cuDisp, self.cuGrid_y, cv2.INTER_LINEAR, stream=self.stream)  #returns a uint8
                elif self.orientation == 'v':
                    self.cuResult = cv2.cuda.remap(self.cuRecUv, self.cuGrid_x, self.cuDisp, cv2.INTER_LINEAR, stream=self.stream)  #returns a uint8
                
                ############### Rectify the UV, RGB and Mask images ########################################### Both rectify and remap sections take around 5ms

                ###############   Mix the MASK and the rectified RGB and UV images ######################## #This section takes 6 to 10 ms (float conversion 25ms)
                self.recMask = self.cuRecMask.download()
                resu = self.cuResult.download()
                resu[:, :, 3] = cv2.convertScaleAbs(self.recMask, alpha=255)
                self.cuResult.upload(resu)
                
                self.cuFinal = cv2.cuda.alphaComp(self.cuResult, self.cuRecColor, cv2.cuda.ALPHA_OVER, stream=self.stream )
                ###############   Mix the MASK and the rectified RGB and UV images ######################## #This section takes 6 to 10 ms (with floats 25ms)

                ############### CUT THE WARP DISTORTION IN THE FINAL IMAGE ###################################
                self.cuFinal = cv2.cuda.warpPerspective(self.cuFinal, self.matrices.cutMatrixUv, (self.resX,self.resY), stream=self.stream)
                ############### CUT THE WARP DISTORTION IN THE FINAL IMAGE ###################################
                    
                final = cv2.convertScaleAbs(self.cuFinal.download())

                cv2.imshow('final', final)
                key = cv2.waitKey(1)

                gpuTime = (time.time_ns() - start) / 1000000
                # print ("GPU Time: " + str(gpuTime))

                if key & 0xFF == ord('q'):
                    break
        # finally:
        #     print("error in gpu thread")
        #     return 