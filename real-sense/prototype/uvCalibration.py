import cv2 as cv
import math
from math import pi
import numpy as np
from matplotlib import pyplot as plt

class myCalib:

    # fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
    #         "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];

    def __init__(self):
        self.cachedMatrix = None
        self.R = None
        self.T = None
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q =  None
        self.M1 = None
        self.D1 = None
        self.M2 = None
        self.D2 = None
        self.mapx1 = None 
        self.mapy1 = None
        self.mapx2 = None
        self.mapy2 = None
        self.height = None
        self.width = None
        self.Vroi = []

    def loadCalibMatrixFile(self, fExt, fInt, h, w):
        self.height = h
        self.width = w

        #Load extrinsic matrix variables 
        fext = cv.FileStorage(fExt, cv.FILE_STORAGE_READ) 
        self.R = fext.getNode("R").mat()
        self.T = fext.getNode("T").mat()
        self.R1 = fext.getNode("R1").mat()
        self.R2 = fext.getNode("R2").mat()
        self.P1 = fext.getNode("P1").mat()
        self.P2 = fext.getNode("P2").mat()

        #Load intrinsic matrix variables 
        fint = cv.FileStorage(fInt, cv.FILE_STORAGE_READ) 
        self.M1 = fint.getNode("M1").mat()  #cameraMatrix[0]
        self.D1 = fint.getNode("D1").mat()  #distCoeffs[0]
        self.M2 = fint.getNode("M2").mat()  #cameraMatrix[1]
        self.D2 = fint.getNode("D2").mat()  #distCoeffs[2]

        n = fext.getNode("validRoi1")
        m = fext.getNode("validRoi2")
        vroi1 = []
        vroi2 = []
        for i in range(n.size()):
            vroi1.append(int(n.at(i).real()))  #x,y,w,h
            vroi2.append(int(m.at(i).real()))  #x,y,w,h

        #get the largest rectangle cut 
        self.Vroi.append(min(vroi1[0],vroi2[0])) #x
        self.Vroi.append(min(vroi1[1],vroi2[1])) #y
        self.Vroi.append(max(vroi1[2],vroi2[2])) #w
        self.Vroi.append(max(vroi1[3],vroi2[3])) #h
        
        fint.release()
        fext.release()

    def cutImage(self, left, right, depth):
        return left[self.Vroi[1]:self.Vroi[1]+self.Vroi[3], self.Vroi[0]:self.Vroi[0]+self.Vroi[2]],  right[self.Vroi[1]:self.Vroi[1]+self.Vroi[3], self.Vroi[0]:self.Vroi[0]+self.Vroi[2]], depth[self.Vroi[1]:self.Vroi[1]+self.Vroi[3], self.Vroi[0]:self.Vroi[0]+self.Vroi[2]] 

    def undistort(self):
        # PROBAR ESTA OPCION !!! OPCIONAL 1!!
        # https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
        # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        #Mat rmap[2][2];
        self.mapx1 = np.ndarray(shape=(self.height, self.width, 1), dtype='float32')
        self.mapy1 = np.ndarray(shape=(self.height, self.width, 1), dtype='float32')
        self.mapx2 = np.ndarray(shape=(self.height, self.width, 1), dtype='float32')
        self.mapy2 = np.ndarray(shape=(self.height, self.width, 1), dtype='float32')

        self.mapx1, self.mapy1 = cv.initUndistortRectifyMap(self.M1, self.D1, self.R1, self.P1,(self.width, self.height), cv.CV_16SC2)
        self.mapx2, self.mapy2 = cv.initUndistortRectifyMap(self.M2, self.D2, self.R2, self.P2,(self.width, self.height), cv.CV_16SC2)

    def remap(self, left, right, depth):  #left is infra, right is normal 
        #remap(mg, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR)
        recRight = np.full_like(right, 0, dtype="uint8")
        recLeft = np.full_like(left, 0, dtype="uint8")
        recDepth = np.full_like(left, 0, dtype="uint8")

        recLeft = cv.remap(left, self.mapx1, self.mapy1, cv.INTER_LINEAR)
        recRight = cv.remap(right, self.mapx2, self.mapy2, cv.INTER_LINEAR) 
        recDepth = cv.remap(left, self.mapx1, self.mapy1, cv.INqTER_LINEAR)
        #recRight = cv.warpPerspective(recRight, self.R1, (self.width, self.height))

        return recLeft, recRight, recDepth  #no cut 
        # return self.cutImage(recLeft, recRight, recDepth)  #cut images

    

    #eliminates distance between the cameras
    #front = infra, back = real photo
    def translate(self, front):
        # T = np.float32([[1, 0, 0], [0, 1, -9]])   #Only translate on Y (CHECK THIS PROBLEM LATER !!!) Original for slack video with rectification PACK1
        # T = np.float32([[1, 0, 10], [0, 1, -10]])   #Only translate on Y (CHECK THIS PROBLEM LATER !!!)
        # T = np.float32([[1, 0, -10], [0, 1, 4]])   #For adjust feature alignment results only DEBUG
        # T = np.float32([[1, 0, 0], [0, 1, 4]])   #Translate infrared image on Y (PACK3)

        T = np.float32([[1, 0, -12], [0, 1, 4]]) # with camera calibration parameters
        front = cv.warpAffine(front, T, (front.shape[1],front.shape[0]))

        return front 

    def siftHomography(self, img1, img2):
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        MIN_MATCH_COUNT = 10
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w,c = img1.shape

            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)
            self.cachedMatrix = M 
            img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
        else:
            print( "Not enough matches were found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None
            self.cachedMatrix = None
        
        print ("[INFO] Homography matrix created.")            
        
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
        img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        plt.imshow(img3, 'gray'),plt.show()

        return self.cachedMatrix

   


