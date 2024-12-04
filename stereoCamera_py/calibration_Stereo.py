import cv2
import numpy as np
import glob

# 设置棋盘格的尺寸
chessboard_size = (8, 5)  # 棋盘格内角点数目，列数8，行数5
# square_size = 1.0  # 每个小方格的实际尺寸（单位可以是米，厘米等）

# 设置对象点的世界坐标系（3D），假设棋盘格的角点在z=0平面上
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) #* square_size

# 用于存储标定数据
obj_points = []  # 3D点
img_points_left = []  # 左相机图像上的2D点
img_points_right = []  # 右相机图像上的2D点

# 获取所有图像文件路径
image_files = sorted(glob.glob('stereoImage/*_L.jpg'))  # 只选择左右视图文件夹下的左相机图像

# 遍历每一对左右图像
for i in range(len(image_files)):
    # 读取左右视图图像
    img_left = cv2.imread(image_files[i])
    img_right = cv2.imread(image_files[i].replace("_L", "_R"))

    # 将图像转为灰度图，进行角点检测
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left and ret_right:
        # 添加3D点和2D点
        obj_points.append(objp)

        # 提升角点精度
        cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), 
                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), 
                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # 添加图像点
        img_points_left.append(corners_left)
        img_points_right.append(corners_right)

# 加载相机的内参
calib_data = np.load("camera_calibration_data.npz")
mtx_left = calib_data['mtx_left']
dist_left = calib_data['dist_left']
mtx_right = calib_data['mtx_right']
dist_right = calib_data['dist_right']

# 双目标定
ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_left, img_points_right, mtx_left, dist_left,
    mtx_right, dist_right, gray_left.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC
)
print(ret)


# 输出结果：保存外参矩阵
print("Rotation Matrix:\n", R)
print("Translation Vector:\n", T)

# 保存外参数据
np.savez("stereo_calibration_data.npz", R=R, T=T)


#############################################################
#                                                           #
#                      左右相机外参                          #
#                                                           #
#############################################################

# 1.4183907115416372
# Rotation Matrix:
#  [[ 0.99930702  0.00131184 -0.03719881]
#  [-0.0012313   0.99999685  0.002188  ]
#  [ 0.03720156 -0.00214068  0.99930549]]
# Translation Vector:
#  [[-4.0009615 ]
#  [ 0.00640141]
#  [-0.20155323]]