
### OLD ALPHA MERGE ALGORITHM
if self.side == 'l':
    self.recMask = self.cuRecMask_left.download(stream=self.stream)
    resu = self.cuResult_left.download(stream=self.stream)
    resu[:, :, 3] = cv2.subtract(self.recMask, self.alpha)
    self.cuResult_left.upload(resu, stream=self.stream)
    
    self.cuFinal_left = cv2.cuda.alphaComp(self.cuResult_left, self.cuRecColor_left, cv2.cuda.ALPHA_OVER, stream=self.stream )

    ############### CUT THE WARP DISTORTION IN THE FINAL IMAGE ###################################
    # self.cuFinal_left = cv2.cuda.warpPerspective(self.cuFinal_left, self.matrices_left.cutMatrixUv, (self.resX,self.resY), stream=self.stream)
    ############### CUT THE WARP DISTORTION IN THE FINAL IMAGE ###################################