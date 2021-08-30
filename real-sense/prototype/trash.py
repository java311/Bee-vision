#?????????????????????????grab cut algorithm//////////////////////////////////////////////////////////////

def gen_element(erosion_size):
    return cv.getStructuringElement(cv.MORPH_RECT,(erosion_size + 1, erosion_size + 1), (erosion_size, erosion_size))

erosion_size = 3
erode_less = gen_element(erosion_size)
erode_more = gen_element(erosion_size * 2)

# The following operation is taking grayscale image,
# performs threashold on it, closes small holes and erodes the white area
def create_mask_from_depth(depth, thresh, tipo):
    depth = cv.threshold(depth, thresh, 255, tipo)
    depth = cv.dilate(depth, erode_less)
    depth = cv.erode(depth, erode_more)
    return depth

# +++++++++++++++ filter near from far objects +++++++++++++++++ (with precision, I hope....)
        #  Generate "near" mask image:
        near = cv.cvtColor(depth_image, cv.BGR2GRAY)
        #  Take just values within range [180-255]
        #  These will roughly correspond to near objects due to histogram equalization
        near = create_mask_from_depth(near, 180, cv.THRESH_BINARY)

        # Generate "far" mask image:
        far = cv.cvtColor(aligned_uv, cv.COLOR_BGR2GRAY)
        far[far == 0] = 255
        # far.setTo(255, far == 0); # Note: 0 value does not indicate pixel near the camera, and requires special attention 
        far = create_mask_from_depth(far, 100, cv.THRESH_BINARY_INV)

        mask = np.fill((resY,resX,1),cv.GC_BGD, dtype=np.int8)
        mask[far == 0] = cv.GC_PR_BGD
        mask[near == 255] = cv.GC_FGD

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (0,0,0,0)
        cv2.grabCut(aligned_uv,mask,rect,bgdModel,fgdModel, 1,GC_INIT_WITH_MASK)

        # Extract foreground pixels based on refined mask from the algorithm
        Mat3b foreground = Mat3b::zeros(color_mat.rows, color_mat.cols)
        color_mat.copyTo(foreground, (mask == GC_FGD) | (mask == GC_PR_FGD))
        imshow(window_name, foreground)
        










        uv_x = infra_image.shape[1] * 1.3
        uv_y = infra_image.shape[0] * 1.3
        uv_x_zero = infra_image.shape[1] - uv_x
        uv_y_zero = infra_image.shape[0] - uv_y

        pts1 = np.float32([[0,0], [infra_image.shape[1],0], [infra_image.shape[1],infra_image.shape[0]], [0,infra_image.shape[0]] ])
        pts2 = np.float32([[0-uv_x_zero,0-uv_y_zero], [uv_x,0-uv_y_zero], [uv_x,uv_y], [0-uv_x_zero,uv_y] ])

        M = cv.getPerspectiveTransform(pts1,pts2)
        infra_image = cv.warpPerspective(infra_image,M,(resX,resY))







