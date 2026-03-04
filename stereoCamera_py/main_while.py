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


def process_stereo_images(img_left_path, img_right_path, output_dir='.', use_wls=True, show_visualization=True):
    """
    处理立体图像对，生成深度图和点云
    
    Args:
        img_left_path: 左相机图像路径
        img_right_path: 右相机图像路径
        output_dir: 输出目录
        use_wls: 是否使用WLS滤波
        show_visualization: 是否显示可视化结果
    
    Returns:
        point_cloud_path: 保存的点云文件路径
    """
    import os
    
    print(f"[INFO] 读取图像: {img_left_path}, {img_right_path}")
    
    # 检查文件存在性
    if not os.path.exists(img_left_path) or not os.path.exists(img_right_path):
        print(f"[ERROR] 图像文件不存在!")
        return None
    
    # 读取图像
    imgl = cv.imread(img_left_path)
    imgr = cv.imread(img_right_path)
    
    if imgl is None or imgr is None:
        print("[ERROR] 无法读取图像文件")
        return None
    
    print(f"[INFO] 图像大小: {imgl.shape}")
    
    # 获取图像尺寸
    high, wide = imgl.shape[0:2]
    
    # 初始化相机参数
    config = stereoCamera()
    print("[INFO] 相机参数已加载")
    
    # 消除图像畸变
    print("[INFO] 正在进行图像畸变矫正...")
    imgl_qb = cv.undistort(imgl, config.cam_matrix_l, config.distortion_l)
    imgr_qb = cv.undistort(imgr, config.cam_matrix_r, config.distortion_r)
    
    # 极线校正
    print("[INFO] 正在进行极线校正...")
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(high, wide, config)
    imgl_jx, imgr_jx = rectifyImage(imgl_qb, imgr_qb, map1x, map1y, map2x, map2y)
    
    # 转换为灰度图像并进行直方图均衡化
    print("[INFO] 正在进行图像预处理...")
    imgl_hd = cv.cvtColor(imgl_jx, cv.COLOR_BGR2GRAY)
    imgr_hd = cv.cvtColor(imgr_jx, cv.COLOR_BGR2GRAY)
    imgl_hd = cv.equalizeHist(imgl_hd)
    imgr_hd = cv.equalizeHist(imgr_hd)
    
    # 立体匹配（SGBM算法）
    print("[INFO] 正在计算视差图 (SGBM)...")
    dispL = opencv_SGBM(imgl_hd, imgr_hd, use_wls=use_wls)
    
    # 视差图空洞填充
    print("[INFO] 正在填充视差图空洞...")
    dispL = fill_disparity_map(dispL)
    
    # 生成深度图
    print("[INFO] 正在生成深度图...")
    depthmap = creatDepthView(dispL)
    
    # 保存深度图
    depth_output = os.path.join(output_dir, 'depth_map.png')
    cv.imwrite(depth_output, depthmap)
    print(f"[INFO] 深度图已保存: {depth_output}")
    
    # 保存视差图
    dispL_vis = cv.normalize(dispL, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    disparity_output = os.path.join(output_dir, 'disparity_map.png')
    cv.imwrite(disparity_output, dispL_vis)
    print(f"[INFO] 视差图已保存: {disparity_output}")
    
    # 生成三维点云
    print("[INFO] 正在生成三维点云...")
    assert imgl.shape[:2] == dispL.shape[:2], "[ERROR] 图像和视差图尺寸不匹配!"
    points, colors = create_point_cloud(dispL, imgl_jx, Q)
    
    # 保存点云
    depth_range = [500, 20000]  # 调整深度范围
    point_cloud_path = os.path.join(output_dir, 'point_cloud.xyz')
    save_point_cloud(points, colors, point_cloud_path, depth_range)
    print(f"[INFO] 点云已保存: {point_cloud_path}")
    print(f"[INFO] 点云包含 {len(points)} 个点")
    
    # 显示可视化结果（Mac上需要额外处理）
    if show_visualization:
        print("[INFO] 显示可视化结果...")
        try:
            cv.imshow('原始图像', cv.resize(imgl, (960, 540)))
            cv.imshow('视差图', cv.resize(dispL_vis, (960, 540)))
            cv.imshow('深度图', cv.resize(depthmap, (960, 540)))
            print("[INFO] 按任意键关闭窗口...")
            cv.waitKey(0)
            cv.destroyAllWindows()
        except Exception as e:
            print(f"[WARNING] 无法显示图像窗口: {e}")
    
    return point_cloud_path


if __name__ == '__main__':
    import os
    import sys
    
    print("=" * 60)
    print("双目立体相机处理系统")
    print("=" * 60)
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义图像路径（相对于脚本目录）
    img_left = os.path.join(script_dir, 'imagefolder', '5_l.jpg')
    img_right = os.path.join(script_dir, 'imagefolder', '5_r.jpg')
    
    # 尝试备选路径
    if not os.path.exists(img_left) or not os.path.exists(img_right):
        print("[WARNING] 默认图像不存在，尝试其他路径...")
        img_left = os.path.join(script_dir, 'imagefolder', 'im0.png')
        img_right = os.path.join(script_dir, 'imagefolder', 'im1.png')
    
    print(f"[INFO] 脚本目录: {script_dir}")
    print(f"[INFO] 输出目录: {script_dir}")
    print()
    
    # 处理立体图像对
    result = process_stereo_images(
        img_left, 
        img_right, 
        output_dir=script_dir,
        use_wls=True,
        show_visualization=True
    )
    
    if result:
        print()
        print("=" * 60)
        print("[SUCCESS] 处理完成!")
        print(f"[INFO] 点云文件位置: {result}")
        print("=" * 60)
        sys.exit(0)
    else:
        print()
        print("=" * 60)
        print("[ERROR] 处理失败")
        print("=" * 60)
        sys.exit(1)

