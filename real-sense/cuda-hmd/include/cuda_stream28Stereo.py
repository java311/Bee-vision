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
    matrices_left = None
    matrices_right = None
    orientation = 'h' 
    add = True   
    side = 'right'
    alpha = 255
    bee_alpha = 255
    beeVision = False 

    ##IMAGES
    color_image = None
    uv_image = None
    depth_image = None

    #GPU VARIABLES
    ####IMAGES
    # cuDepth = cv2.cuda_GpuMat()
    cuUv = cv2.cuda_GpuMat() 
    cuDisp_left = cv2.cuda_GpuMat()
    cuDisp_right = cv2.cuda_GpuMat()
    ####MAPPING
    cuMapx1_left = cv2.cuda_GpuMat()
    cuMapy1_left = cv2.cuda_GpuMat()
    cuMapx2_left = cv2.cuda_GpuMat()
    cuMapy2_left = cv2.cuda_GpuMat()
    cuMapx1_right = cv2.cuda_GpuMat()
    cuMapy1_right = cv2.cuda_GpuMat()
    cuMapx2_right = cv2.cuda_GpuMat()
    cuMapy2_right = cv2.cuda_GpuMat()

    ones_f = cv2.cuda_GpuMat()
    bases_left = cv2.cuda_GpuMat()
    bases_right = cv2.cuda_GpuMat()

    cuGrid_x = cv2.cuda_GpuMat()
    cuGrid_y = cv2.cuda_GpuMat()
    ####RECTIFY AND REMAP BLOCK
    cuRecColor_left = cv2.cuda_GpuMat()
    cuRecUv_left = cv2.cuda_GpuMat()
    cuRecMask_left = cv2.cuda_GpuMat()
    cuRecColor_right = cv2.cuda_GpuMat()
    cuRecUv_right = cv2.cuda_GpuMat()
    cuRecMask_right = cv2.cuda_GpuMat()

    cuResult_left = cv2.cuda_GpuMat()
    cuResult_right = cv2.cuda_GpuMat()
    cuFinal_left = cv2.cuda_GpuMat()
    cuFinal_right = cv2.cuda_GpuMat()

    final_left = None
    final_right = None

    ones = cv2.cuda_GpuMat()
    # MASKING
    cuMask = cv2.cuda_GpuMat()
    cuClip1 = cv2.cuda_GpuMat()
    cuClip2 = cv2.cuda_GpuMat()
    cuMaskDist = cv2.cuda_GpuMat()
    cuMaskZeros = cv2.cuda_GpuMat()
    alphas = cv2.cuda_GpuMat()
    cu_r_por = cv2.cuda_GpuMat()
    cu_g_por = cv2.cuda_GpuMat()
    cu_b_por = cv2.cuda_GpuMat()
    cu_bee_alpha = cv2.cuda_GpuMat()
    cuBeeColor_left = cv2.cuda_GpuMat()
    cuBeeColor_right = cv2.cuda_GpuMat()
    cuErosionFilter = None
    cuDilateFilter = None
    cuGaussFilter = None

    def __init__(self, stream, resX, resY, alphaValue, matrices_left, matrices_right, base_rgb_uv_left, base_rgb_uv_right, clip_dist, orientation, add, event, side):
        # GPU MATRICES INIT
        self.stream = stream
        self.resX = resX
        self.resY = resY
        self.matrices_left = matrices_left
        self.matrices_right = matrices_right
        self.event = event
        self.orientation = orientation
        self.add = add
        self.side = side
        self.base_rgb_uv_left = base_rgb_uv_left
        self.base_rgb_uv_right = base_rgb_uv_right
        self.alpha = 255 - alphaValue
        self.bee_alpha = 255 - 153
        self.beeVision = True

        ##########################  PINNED MEMORY  ########################################
        self.cuDisp_left.upload(np.full((resY, resX), 0, dtype="float32"), stream=self.stream)  #this static init may cause a bug
        self.cuDisp_right.upload(np.full((resY, resX), 0, dtype="float32"), stream=self.stream)  #this static init may cause a bug

        
        self.cuMapx1_left.upload(matrices_left.mapx1_uv, stream=self.stream)  #these are static
        self.cuMapy1_left.upload(matrices_left.mapy1_uv, stream=self.stream)  #these are static 
        self.cuMapx2_left.upload(matrices_left.mapx2_uv, stream=self.stream)  #these are static
        self.cuMapy2_left.upload(matrices_left.mapy2_uv, stream=self.stream)  #these are static

        self.cuMapx1_right.upload(matrices_right.mapx1_uv, stream=self.stream)  #these are static
        self.cuMapy1_right.upload(matrices_right.mapy1_uv, stream=self.stream)  #these are static
        self.cuMapx2_right.upload(matrices_right.mapx2_uv, stream=self.stream)  #these are static
        self.cuMapy2_right.upload(matrices_right.mapy2_uv, stream=self.stream)  #these are static

        ##########################  PINNED MEMORY  ########################################

        # self.ones_f = cv2.cuda_GpuMat();  
        self.ones_f.upload(np.full((resY, resX), 1.0, dtype="float32"), stream=self.stream)          #static
        # self.bases = cv2.cuda_GpuMat();  
        self.bases_left.upload(np.full((resY, resX), self.base_rgb_uv_left, dtype="float32"), stream=self.stream)    #static
        self.bases_right.upload(np.full((resY, resX), self.base_rgb_uv_right, dtype="float32"), stream=self.stream)    #static

        grid = np.indices((resY, resX))        #static
        # self.cuGrid_x = cv2.cuda_GpuMat()  
        self.cuGrid_x.upload(grid[1].astype(np.float32), stream=self.stream)        #static
        # self.cuGrid_y = cv2.cuda_GpuMat()  
        self.cuGrid_y.upload(grid[0].astype(np.float32), stream=self.stream)        #static


        # Rectify and remap block 
        # self.cuRecColor = cv2.cuda_GpuMat(); 
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
        # self.ones = cv2.cuda_GpuMat()  
        self.ones.upload(np.full((resY, resX, 3), 1, dtype="float32"), stream=self.stream )

        #######  MASKING
        # self.cuMask = cv2.cuda_GpuMat()
        self.cuMask.upload(np.full((resY, resX), 0, dtype="uint16"), stream=self.stream )
        # self.cuClip1 = cv2.cuda_GpuMat()  
        self.cuClip1.upload(np.full((resY, resX), 0, dtype="uint8"), stream=self.stream )
        # self.cuClip2 = cv2.cuda_GpuMat()  
        self.cuClip2.upload(np.full((resY, resX), 0, dtype="uint8"), stream=self.stream )
        # self.cuMaskDist = cv2.cuda_GpuMat() 
        self.cuMaskDist.upload(np.full((resY, resX), int(clip_dist), dtype="uint16"), stream=self.stream )
        # self.cuMaskZeros = cv2.cuda_GpuMat() 
        self.cuMaskZeros.upload(np.full((resY, resX), 0, dtype="uint16"), stream=self.stream )

        self.alphas.upload(np.full((resY, resX), self.alpha, dtype="uint8"), stream=self.stream )

        self.cu_r_por.upload(np.full((resY, resX), int(1/0.40), dtype="uint8"), stream=self.stream )
        self.cu_g_por.upload(np.full((resY, resX), int(1/0.40), dtype="uint8"), stream=self.stream )
        self.cu_b_por.upload(np.full((resY, resX), int(1/0.10), dtype="uint8"), stream=self.stream )

        self.cu_bee_alpha.upload(np.full((resY, resX), self.bee_alpha, dtype="uint8"), stream=self.stream )

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


    def updateImages(self, color_image, uv_image, depth_image, side):
        
        self.color_image = color_image
        self.uv_image = uv_image
        self.depth_image = depth_image
        self.side = side
        # self.matrices_left = matrices_left
        # self.matrices_right = matrices_right
        # self.base_rgb_uv_left = base_rgb_uv_left
        # self.base_rgb_uv_right = base_rgb_uv_right

        # #######################  UPDATE THE STATIC GPU MATRICES #############################
        # self.bases_left.upload(np.full((self.resY, self.resX), base_rgb_uv_left, dtype="float32"), stream=self.stream)
        # self.bases_right.upload(np.full((self.resY, self.resX), base_rgb_uv_right, dtype="float32"), stream=self.stream)
        # self.cuMapx1_uv.upload(matrices_left.mapx1_uv, stream=self.stream)  #these are static
        # self.cuMapy1_uv.upload(matrices_left.mapy1_uv, stream=self.stream)  #these are static
        # self.cuMapx2_uv.upload(matrices_left.mapx2_uv, stream=self.stream)  #these are static
        # self.cuMapy2_uv.upload(matrices_left.mapy2_uv, stream=self.stream)  #these are static
        # self.cuMapx1_uv.upload(matrices_right.mapx1_uv, stream=self.stream)  #these are static
        # self.cuMapy1_uv.upload(matrices_right.mapy1_uv, stream=self.stream)  #these are static
        # self.cuMapx2_uv.upload(matrices_right.mapx2_uv, stream=self.stream)  #these are static
        # self.cuMapy2_uv.upload(matrices_right.mapy2_uv, stream=self.stream)  #these are static
        # #######################  UPDATE THE CONSTANT GPU MATRICES #############################

    def run(self):
        # try:
        while True:
            event_is_set = self.event.wait()
            if event_is_set:
                start = time.time_ns()
                # ##################### Remove background  GPU ################################  17ms   
                # mask = np.where((depth_image > clipping_distance) | (depth_image <= 0), 0.0, alphaValue)   # orignal formula	
                self.cuMask.upload(np.array(self.depth_image), stream=self.stream)
                self.cuClip1 = cv2.cuda.compare(self.cuMask, self.cuMaskDist, cv2.CMP_LE, stream=self.stream)   #depth < clipping_distance (inverted)
                self.cuClip2 = cv2.cuda.compare(self.cuMask, self.cuMaskZeros, cv2.CMP_GT, stream=self.stream)  #depth >= 0   (inverted)
                self.cuMask = cv2.cuda.bitwise_and(self.cuClip1, self.cuClip2, stream=self.stream)   #bitwise OR
                
                self.cuMask = self.cuErosionFilter.apply(self.cuMask, stream=self.stream)
                self.cuMask = self.cuDilateFilter.apply(self.cuMask, stream=self.stream)
                self.cuMask = self.cuGaussFilter.apply(self.cuMask, stream=self.stream)
                # ##################### Remove background  GPU ################################ 13ms

                ############### GET DISPARITY GPU for UV CAMERA ########################################### This step takes 5ms 
        
                # Make all the uploads necessary (depth, mapx) 
                

                #Remap of the depth map
                if self.side == 'l':
                    self.cuDisp_left.upload(self.depth_image.astype(np.float32), stream=self.stream)  #CHECK THIS, THE CAST WAS ORIGINALLY AFTER THE REMAP
                    self.cuDisp_left = cv2.cuda.remap(self.cuDisp_left, self.cuMapx1_left, self.cuMapy1_left, cv2.INTER_LINEAR, stream=self.stream)
                    self.cuDisp_left = cv2.cuda.divide(self.ones_f, self.cuDisp_left, stream=self.stream)
                    self.cuDisp_left = cv2.cuda.multiply(self.cuDisp_left, self.bases_left, stream=self.stream)
                else:
                    self.cuDisp_right.upload(self.depth_image.astype(np.float32), stream=self.stream)  #CHECK THIS, THE CAST WAS ORIGINALLY AFTER THE REMAP
                    self.cuDisp_right = cv2.cuda.remap(self.cuDisp_right, self.cuMapx1_right, self.cuMapy1_right, cv2.INTER_LINEAR, stream=self.stream)
                    self.cuDisp_right = cv2.cuda.divide(self.ones_f, self.cuDisp_right, stream=self.stream)
                    self.cuDisp_right = cv2.cuda.multiply(self.cuDisp_right, self.bases_right, stream=self.stream)

                # print (str(self.side) + " : " +  str(np.mean(self.bases.download()) ) )

                #Transform the depth map to disparity map aligned to the external camera 
                if self.orientation == 'h':
                    if self.add:
                        if self.side == 'l':
                            self.cuDisp_left = cv2.cuda.add(self.cuGrid_x, self.cuDisp_left, stream=self.stream)
                        else:
                            self.cuDisp_right = cv2.cuda.add(self.cuGrid_x, self.cuDisp_right, stream=self.stream)
                    else:
                        if self.side == 'l':
                            self.cuDisp_left = cv2.cuda.subtract(self.cuGrid_x, self.cuDisp_left, stream=self.stream)
                        else:
                            self.cuDisp_right = cv2.cuda.subtract(self.cuGrid_x, self.cuDisp_right, stream=self.stream)
                elif self.orientation == 'v':
                    if self.add:
                        if self.side == 'l':
                            self.cuDisp_left = cv2.cuda.add(self.cuGrid_y, self.cuDisp_left, stream=self.stream)
                        else:
                            self.cuDisp_right = cv2.cuda.add(self.cuGrid_y, self.cuDisp_right, stream=self.stream)
                    else:
                        if self.side == 'l':
                            self.cuDisp_left = cv2.cuda.subtract(self.cuGrid_y, self.cuDisp_left, stream=self.stream)
                        else:
                            self.cuDisp_right = cv2.cuda.subtract(self.cuGrid_y, self.cuDisp_right, stream=self.stream)
                
                ############### GET DISPARITY GPU for UV CAMERA ###########################################

                ############### Rectify the UV, RGB and Mask images ########################################### Both rectify and remap sections take around 5ms
        
                self.cuUv.upload(self.uv_image, stream=self.stream)
                
                if self.side == 'l':
                    self.cuRecColor_left.upload(self.color_image, stream=self.stream)
                    self.cuRecColor_left = cv2.cuda.remap(self.cuRecColor_left, self.cuMapx1_left, self.cuMapy1_left, cv2.INTER_LINEAR, stream=self.stream)  #rgb
                    self.cuRecUv_left = cv2.cuda.remap(self.cuUv, self.cuMapx2_left, self.cuMapy2_left, cv2.INTER_LINEAR, stream=self.stream) #uv
                    self.cuRecMask_left = cv2.cuda.remap(self.cuMask, self.cuMapx1_left, self.cuMapy1_left, cv2.INTER_LINEAR, stream=self.stream) #mask aligned to rgb
                else:
                    self.cuRecColor_right.upload(self.color_image, stream=self.stream)
                    self.cuRecColor_right = cv2.cuda.remap(self.cuRecColor_right, self.cuMapx1_right, self.cuMapy1_right, cv2.INTER_LINEAR, stream=self.stream)  #rgb
                    self.cuRecUv_right = cv2.cuda.remap(self.cuUv, self.cuMapx2_right, self.cuMapy2_right, cv2.INTER_LINEAR, stream=self.stream) #uv
                    self.cuRecMask_right = cv2.cuda.remap(self.cuMask, self.cuMapx1_right, self.cuMapy1_right, cv2.INTER_LINEAR, stream=self.stream) #mask aligned to rgb

                ###############   Make the remaping of the UV image over the RGB image using the computed disparity Map
                if self.orientation == 'h':
                    if self.side == 'l':
                        self.cuResult_left = cv2.cuda.remap(self.cuRecUv_left, self.cuDisp_left, self.cuGrid_y, cv2.INTER_LINEAR, stream=self.stream)  #returns a uint8
                    else:
                        self.cuResult_right = cv2.cuda.remap(self.cuRecUv_right, self.cuDisp_right, self.cuGrid_y, cv2.INTER_LINEAR, stream=self.stream)  #returns a uint8
                elif self.orientation == 'v':
                    if self.side == 'l':
                        self.cuResult_left = cv2.cuda.remap(self.cuRecUv_left, self.cuGrid_x, self.cuDisp_left, cv2.INTER_LINEAR, stream=self.stream)  #returns a uint8
                    else:
                        self.cuResult_right = cv2.cuda.remap(self.cuRecUv_right, self.cuGrid_x, self.cuDisp_right, cv2.INTER_LINEAR, stream=self.stream)  #returns a uint8
                ############### Rectify the UV, RGB and Mask images ########################################### Both rectify and remap sections take around 5ms

                # if self.side == 'l':
                #     pinchi1 =  self.cuRecColor_left.download(stream=self.stream)
                #     pinchi2 =  self.cuRecUv_left.download(stream=self.stream)
                #     pinchi3 =  self.cuRecMask_left.download(stream=self.stream)
                #     cv2.imshow(str(self.side) + '_rgb', pinchi1)
                #     cv2.imshow(str(self.side) + '_center', pinchi2)
                #     cv2.imshow(str(self.side) + '_mask', pinchi3)

                ###############   Mix the MASK and the rectified RGB and UV images ######################## #This section takes 6 to 10 ms (float conversion 25ms)
                if self.side == 'l':
                    if self.beeVision:
                        colorSplit = cv2.cuda.split(self.cuRecColor_left, stream=self.stream)
                        uvSplit = cv2.cuda.split(self.cuResult_left, stream=self.stream)
                        colorSplit[2] = colorSplit[1]  #green to red 
                        colorSplit[1] = colorSplit[0]  #blue to green
                        colorSplit[0].setTo(0, stream=self.stream) #blue to 0
                        
                        uvSplit[1] = cv2.cuda.divide( uvSplit[1], self.cu_g_por, stream=self.stream) # green 40%
                        uvSplit[2] = cv2.cuda.divide( uvSplit[2], self.cu_r_por, stream=self.stream) # red 40%
                        uvSplit[0] = cv2.cuda.divide( uvSplit[0], self.cu_b_por, stream=self.stream) # blue 20%
                        uvSplit[0] = cv2.cuda.add(cv2.cuda.add(uvSplit[1], uvSplit[2], stream=self.stream), uvSplit[0], stream=self.stream )
                        uvSplit[1].setTo(0, stream=self.stream)
                        uvSplit[2].setTo(0, stream=self.stream)
                        
                        colorSplit[3] = self.cuRecMask_left      # mask only to the depth area
                        self.cuRecMask_left = cv2.cuda.subtract(self.cuRecMask_left, self.cu_bee_alpha, stream=self.stream) # adjusting alpha
                        uvSplit[3] = self.cuRecMask_left     # replacing alpha for alpha mix

                        color_merge = cv2.cuda.merge(colorSplit, stream=self.stream)
                        self.cuBeeColor_left.upload(color_merge, stream=self.stream)
                        uv_merge = cv2.cuda.merge(uvSplit, stream=self.stream)
                        self.cuResult_left.upload(uv_merge, stream=self.stream)

                        self.cuBeeColor_left = cv2.cuda.alphaComp(self.cuResult_left, self.cuBeeColor_left, cv2.cuda.ALPHA_OVER, stream=self.stream )

                        self.cuFinal_left = cv2.cuda.alphaComp(self.cuBeeColor_left, self.cuRecColor_left, cv2.cuda.ALPHA_OVER, stream=self.stream )
                    else:
                        self.cuRecMask_left = cv2.cuda.subtract(self.cuRecMask_left, self.alphas, stream=self.stream)
                        uvSplit = cv2.cuda.split(self.cuResult_left, stream=self.stream)
                        uvSplit[3] = self.cuRecMask_left     #replacing alpha 
                        uv_merge = cv2.cuda.merge(uvSplit, stream=self.stream)
                        self.cuResult_left.upload(uv_merge, stream=self.stream)

                        self.cuFinal_left = cv2.cuda.alphaComp(self.cuResult_left, self.cuRecColor_left, cv2.cuda.ALPHA_OVER, stream=self.stream )

                    ############### CUT THE WARP DISTORTION IN THE FINAL IMAGE ###################################
                    # self.cuFinal_left = cv2.cuda.warpPerspective(self.cuFinal_left, self.matrices_left.cutMatrixUv, (self.resX,self.resY), stream=self.stream)
                    ############### CUT THE WARP DISTORTION IN THE FINAL IMAGE ###################################
                else:                    
                    if self.beeVision:
                        colorSplit = cv2.cuda.split(self.cuRecColor_right, stream=self.stream)
                        uvSplit = cv2.cuda.split(self.cuResult_right, stream=self.stream)
                        colorSplit[2] = colorSplit[1]  #green to red 
                        colorSplit[1] = colorSplit[0]  #blue to green
                        colorSplit[0].setTo(0, stream=self.stream) #blue to 0
                        
                        uvSplit[1] = cv2.cuda.divide( uvSplit[1], self.cu_g_por, stream=self.stream) # green 40%
                        uvSplit[2] = cv2.cuda.divide( uvSplit[2], self.cu_r_por, stream=self.stream) # red 40%
                        uvSplit[0] = cv2.cuda.divide( uvSplit[0], self.cu_b_por, stream=self.stream) # blue 20%
                        uvSplit[0] = cv2.cuda.add(cv2.cuda.add(uvSplit[1], uvSplit[2], stream=self.stream), uvSplit[0], stream=self.stream )
                        uvSplit[1].setTo(0, stream=self.stream)
                        uvSplit[2].setTo(0, stream=self.stream)
                        
                        colorSplit[3] = self.cuRecMask_right      # mask only to the depth area
                        self.cuRecMask_right = cv2.cuda.subtract(self.cuRecMask_right, self.cu_bee_alpha, stream=self.stream) # adjusting alpha
                        uvSplit[3] = self.cuRecMask_right     # replacing alpha for alpha mix

                        color_merge = cv2.cuda.merge(colorSplit, stream=self.stream)
                        self.cuBeeColor_right.upload(color_merge, stream=self.stream)
                        uv_merge = cv2.cuda.merge(uvSplit, stream=self.stream)
                        self.cuResult_right.upload(uv_merge, stream=self.stream)

                        self.cuBeeColor_right = cv2.cuda.alphaComp(self.cuResult_right, self.cuBeeColor_right, cv2.cuda.ALPHA_OVER, stream=self.stream )

                        self.cuFinal_right = cv2.cuda.alphaComp(self.cuBeeColor_right, self.cuRecColor_right, cv2.cuda.ALPHA_OVER, stream=self.stream )

                    else:
                        self.cuRecMask_right = cv2.cuda.subtract(self.cuRecMask_right, self.alphas, stream=self.stream)
                        uvSplit = cv2.cuda.split(self.cuResult_right, stream=self.stream)
                        uvSplit[3] = self.cuRecMask_right     #replacing alpha 
                        uv_merge = cv2.cuda.merge(uvSplit, stream=self.stream)
                        self.cuResult_right.upload(uv_merge, stream=self.stream)

                        self.cuFinal_right = cv2.cuda.alphaComp(self.cuResult_right, self.cuRecColor_right, cv2.cuda.ALPHA_OVER, stream=self.stream )

                    ############### CUT THE WARP DISTORTION IN THE FINAL IMAGE ###################################
                    # self.cuFinal_qright = cv2.cuda.warpPerspective(self.cuFinal_right, self.matrices_right.cutMatrixUv, (self.resX,self.resY), stream=self.stream)
                    ############### CUT THE WARP DISTORTION IN THE FINAL IMAGE ###################################
                ###############   Mix the MASK and the rectified RGB and UV images ######################## #This section takes 6 to 10 ms (with floats 25ms)

                

                if self.side == 'l':
                    final_left = cv2.convertScaleAbs(self.cuFinal_left.download(stream=self.stream))
                    final_left = cv2.resize(final_left, (int(self.resX/2), int(self.resY/2)), interpolation = cv2.INTER_AREA) #resize for DEBUG
                    cv2.imshow('final-'+str(self.side), final_left)
                    key = cv2.waitKey(1)
                else:
                    final_right = cv2.convertScaleAbs(self.cuFinal_right.download(stream=self.stream))
                    final_right = cv2.resize(final_right, (int(self.resX/2), int(self.resY/2)), interpolation = cv2.INTER_AREA)  #resize for DEBUG
                    cv2.imshow('final-'+str(self.side), final_right)
                    key = cv2.waitKey(1)


                # final = self.cuResult.download() #DEBUG
                # if self.stream is not None:
                #     self.stream.waitForCompletion()  #blocks thread until all the stream operations had been completed

                gpuTime = (time.time_ns() - start) / 1000000
                # print ("GPU Time: " + str(gpuTime))

                if key & 0xFF == ord('q'):
                    break
        # finally:
        #     print("error in gpu thread")
        #     return 