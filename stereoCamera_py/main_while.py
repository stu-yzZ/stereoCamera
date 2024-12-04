import cv2 as cv
import cv2
import numpy as np
from matplotlib import pyplot as plt
from solution import insert_depth_filled,fill_disparity_map



class stereoCamera(object):
    def __init__(self):
        #自己标定的参数
        # # 左相机内参
        # self.cam_matrix_l = np.array(  [[1.59751383e+03, 0.00000000e+00, 9.18974272e+02],
        #                                 [0.00000000e+00, 1.59550531e+03, 4.89658812e+02],
        #                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


        # # 右相机内参
        # self.cam_matrix_r = np.array(   [[1.58434959e+03, 0.00000000e+00, 9.15435874e+02],
        #                                 [0.00000000e+00, 1.58255597e+03, 4.80750700e+02],
        #                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        # self.distortion_l = np.array([ [-5.95533368e-02,  4.08572992e-01,  5.45587698e-04, -2.39821407e-03, -5.91152502e-01]])
        # self.distortion_r = np.array([-0.0472388,   0.2076486,  -0.00080736, -0.00133338, -0.15354566])
        # # 旋转矩阵
        # self.R = np.array(  [   [ 9.99999792e-01, -5.44060961e-04,  3.47312924e-04],
        #                         [ 5.41323609e-04,  9.99969171e-01,  7.83354538e-03],
        #                         [-3.51564143e-04, -7.83335574e-03,  9.99969257e-01]])

        # # 平移矩阵
        # self.T = np.array(  [   [ 3.97026018],
        #                         [-0.00728108],
        #                         [-0.29210538]])

        self.cam_matrix_l = np.array([  [1733.74, 0.0, 792.27],
                                        [0.0, 1733.74, 541.89],
                                        [0., 0., 1.]])
        self.cam_matrix_r = np.array([  [1733.74, 0.0, 792.27],
                                        [0.0, 1733.74, 541.89],
                                        [0., 0., 1.]])
        self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
        self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
        self.R = np.identity(3, dtype=np.float64)
        self.T = np.array([[-536.62], [0.0], [0.0]])
        self.doffs = 0.0
        self.isRectified = True
        

def getRectifyTransform(high, wide, config):
    left_K = config.cam_matrix_l
    right_K = config.cam_matrix_r
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                     (wide, high), R, T, alpha=0)
    map1x, map1y = cv.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (wide, high), cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (wide, high), cv.CV_32FC1)
    return map1x, map1y, map2x, map2y, Q

def rectifyImage(img1, img2, map1x, map1y, map2x, map2y):
    rectify_img1 = cv.remap(img1, map1x, map1y, cv.INTER_AREA)
    rectify_img2 = cv.remap(img2, map2x, map2y, cv.INTER_AREA)
    return rectify_img1, rectify_img2

def draw_line(img1, img2):
    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:img1.shape[0], 0:img1.shape[1]] = img1
    output[0:img2.shape[0], img1.shape[1]:] = img2
    line_interval = 50  # 直线间隔
    for k in range(height // line_interval):
        cv.line(output, (0, line_interval * (k + 1)),
                (2 * width, line_interval * (k + 1)),
                (0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    plt.imshow(output, 'gray')
    plt.show()
    # cv.imwrite('./lineOut.jpg',output)
    return output

def opencv_SGBM(left_img, right_img, use_wls=False):
    blockSize = 11
    paramL = {"minDisparity": 0,              #表示可能的最小视差值。通常为0，但有时校正算法会移动图像，所以参数值也要相应调整
              "numDisparities": 170,          #表示最大的视差值与最小的视差值之差，这个差值总是大于0。在当前的实现中，这个值必须要能被16整除，越大黑色边缘越多，表示不能计算视差的区域
              "blockSize": blockSize,
              "P1": 8 * 3 * blockSize * blockSize,          #控制视差图平滑度的第一个参数
              "P2": 32 * 3 * blockSize * blockSize,         #控制视差图平滑度的第二个参数，值越大，视差图越平滑。P1是邻近像素间视差值变化为1时的惩罚值，
                                                            #p2是邻近像素间视差值变化大于1时的惩罚值。算法要求P2>P1,stereo_match.cpp样例中给出一些p1和p2的合理取值。
              "disp12MaxDiff": 1,            #表示在左右视图检查中最大允许的偏差（整数像素单位）。设为非正值将不做检查。
              "uniquenessRatio": 10,          #表示由代价函数计算得到的最好（最小）结果值比第二好的值小多少（用百分比表示）才被认为是正确的。通常在5-15之间。
              "speckleWindowSize": 50,       #表示平滑视差区域的最大窗口尺寸，以考虑噪声斑点或无效性。将它设为0就不会进行斑点过滤，否则应取50-200之间的某个值。
              "speckleRange": 1,              #指每个已连接部分的最大视差变化，如果进行斑点过滤，则该参数取正值，函数会自动乘以16、一般情况下取1或2就足够了。
              "preFilterCap": 31,
              "mode": cv.STEREO_SGBM_MODE_SGBM_3WAY
              }
    matcherL = cv.StereoSGBM_create(**paramL)
    # 计算视差图
    dispL = matcherL.compute(left_img, right_img)
    
    # WLS滤波平滑优化图像
    if use_wls:
        paramR = paramL
        paramR['minDisparity'] = -paramL['numDisparities']
        matcherR = cv.StereoSGBM_create(**paramR)
        dispR = matcherR.compute(right_img, left_img)
        # dispR = np.int16(dispR)
        lmbda = 80000
        sigma = 1.0
        filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=matcherL)
        filter.setLambda(lmbda)
        filter.setSigmaColor(sigma)
        dispL = filter.filter(dispL, left_img, None, dispR)

    #双边滤波
    dispL = cv2.bilateralFilter(dispL.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)

    # 除以16得到真实视差（因为SGBM算法得到的视差是×16的）
    dispL[dispL < 0] = 1e-6
    dispL = dispL.astype(np.int16)
    dispL = dispL / 16.0

    return dispL


def create_point_cloud(dispL, img, Q):
    # step = 2  # 取样间隔
    # dispL = dispL[::step, ::step]
    # img = img[::step, ::step]
    points = cv.reprojectImageTo3D(dispL, Q)
    colors = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # mask = dispL > dispL.min()
    mask = (dispL > 0)
    points = points.astype(np.float16)
    out_points = points[mask]
    out_colors = colors[mask]
    return out_points, out_colors

def save_point_cloud(points, colors, filename, depth_range):
    colors = colors.reshape(-1, 3)
    points = points.reshape(-1, 3)
    # 只要规定区域的点
    print("Depth range in point cloud:", points[:, 2].min(), points[:, 2].max())
    valid_indices = np.where((points[:, 2] > depth_range[0]) & (points[:, 2] < depth_range[1]))[0]
    filtered_points = points[valid_indices]
    # 带颜色的
    filtered_colors = colors[valid_indices]
    point_cloud = np.hstack([filtered_points, filtered_colors])
    np.savetxt(filename, point_cloud, fmt='%0.4f %0.4f %0.4f %d %d %d')
    # # 不带颜色的
    # point_cloud = filtered_points
    # np.savetxt(filename, point_cloud, fmt='%0.4f %0.4f %0.4f')

def creatDepthView(dispL = None,focal_length=1733.74,baseline = 0.53662): #focal_length=1733.74,baseline = 0.53662
    dispL[dispL < 55] = 55
    dispL[dispL > 142] = 142
    depth_map = (baseline * focal_length) / (dispL.astype(np.float32))

    print(np.max(depth_map))
    print(np.min(depth_map))
    

    # 归一化深度图到 0-255 范围
    depth_normalized = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)

    # 使用伪彩色图显示深度
    depth_color_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    return depth_color_map


if __name__ == '__main__':

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    # if True:
    while True:

        ret, frame = cap.read()
    
        if not ret:
            print("无法读取相机帧")
            break

        h,w = frame.shape[0],frame.shape[1]
        imgl = frame[:h, 0:w//2]
        imgr = frame[:h, w//2:w]

        # 读取图像
        imgl = cv.imread('./imagefolder/5_l.jpg')
        imgr = cv.imread('./imagefolder/5_r.jpg')
        # imgl = cv.imread('./imagefolder/im0.png')
        # imgr = cv.imread('./imagefolder/im1.png')
        high, wide = imgl.shape[0:2]

        # # 读取相机参数
        config = stereoCamera()

        # # 消除图像畸变
        imgl_qb = cv.undistort(imgl, config.cam_matrix_l, config.distortion_l)
        imgr_qb = cv.undistort(imgr, config.cam_matrix_r, config.distortion_r)


        # # 极线校正
        map1x, map1y, map2x, map2y, Q = getRectifyTransform(high, wide, config)
        imgl_jx, imgr_jx = rectifyImage(imgl_qb, imgr_qb, map1x, map1y, map2x, map2y)
        # print("Print Q!")
        # print(Q)

        # # 绘制等间距平行线，检查效果
        line = draw_line(imgl_jx, imgr_jx)

        # # 转换为灰度图像
        imgl_hd = cv.cvtColor(imgl_jx, cv.COLOR_BGR2GRAY)
        imgr_hd = cv.cvtColor(imgr_jx, cv.COLOR_BGR2GRAY)


        imgl_hd = cv.cvtColor(imgl, cv.COLOR_BGR2GRAY)
        imgr_hd = cv.cvtColor(imgr, cv.COLOR_BGR2GRAY)
        imgl_hd = cv.equalizeHist(imgl_hd)
        imgr_hd = cv.equalizeHist(imgr_hd)


        # 立体匹配——SBGM算法
        dispL = opencv_SGBM(imgl_hd, imgr_hd, True)

        # 视差图空洞填充
        dispL = fill_disparity_map(dispL)

        # # 计算视差图中每个值出现的次数  用来判断视差图中有效视差的范围
        # unique, counts = np.unique(dispL, return_counts=True)
        # # 绘制直方图
        # plt.figure(figsize=(10, 6))
        # plt.bar(unique, counts, width=1, color='b', alpha=0.7)
        # plt.xlabel('视差值')
        # plt.ylabel('出现次数')
        # plt.title('视差图直方图')
        # plt.show()

        #深度图
        depthmap = creatDepthView(dispL)

        # 深度图空洞填充  深度图填充代码有问题
        # depthmap = insert_depth_filled(depthmap)

        #show original depthMap distanceMap
        cv.imshow('image',cv.resize(imgl,(960,540)))
        cv.imshow('dispL',cv.resize(dispL,(960,540)))
        cv.imshow('depthmap',cv.resize(depthmap,(960,540)))
        cv.imwrite('depth.png',depthmap)
        cv.imwrite('displ.png',dispL)
        cv.waitKey(0)


        # 生成三维点云
        assert imgl.shape[:2] == dispL.shape[:2], "Image and disparity map must have the same resolution!"
        points, colors = create_point_cloud(dispL, imgl_jx, Q)

        # 保存点云（带颜色.xyz;不带颜色.ply）
        depth_range = [5000, 20000]
        save_point_cloud(points, colors, 'point_cloud1.xyz', depth_range)
        
        # 按下 'q' 键退出循环
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# 释放资源
cap.release()
cv.destroyAllWindows()

