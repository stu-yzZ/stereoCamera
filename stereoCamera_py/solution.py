import cv2
import numpy as np


def insert_depth_filled(depth_map):
    """
    填充深度图中的空洞（深度图为32f类型，单位为米）
    使用类似于积分图的方法进行填充。
    
    :param depth_map: 输入的深度图（32f类型）
    :return: 填充后的深度图
    """
    height, width = depth_map.shape[0],depth_map.shape[1]
    # 创建积分图和点数积分图
    integral_map = np.zeros((height, width), dtype=np.float64)
    pts_map = np.zeros((height, width), dtype=np.int32)
    
    # 填充积分图和点数图
    for i in range(height):
        for j in range(width):
            if depth_map[i, j] > 1e-3:
                integral_map[i, j] = depth_map[i, j]
                pts_map[i, j] = 1

    # 计算积分图
    for i in range(height):
        for j in range(1, width):
            integral_map[i, j] += integral_map[i, j-1]
            pts_map[i, j] += pts_map[i, j-1]

    for i in range(1, height):
        for j in range(width):
            integral_map[i, j] += integral_map[i-1, j]
            pts_map[i, j] += pts_map[i-1, j]

    # 使用窗口大小逐步填充空洞
    wnd = 2
    dWnd = 2.0
    while dWnd > 1:
        wnd = int(dWnd)
        dWnd /= 2
        for i in range(height):
            for j in range(width):
                # 计算邻域范围
                left = max(0, j - wnd - 1)
                right = min(j + wnd, width - 1)
                top = max(0, i - wnd - 1)
                bottom = min(i + wnd, height - 1)

                # 计算积分区域
                dx = right - left
                dy = (bottom - top) * width
                id_left_top = top * width + left
                id_right_top = id_left_top + dx
                id_left_bottom = id_left_top + dy
                id_right_bottom = id_left_bottom + dx

                pts_cnt = pts_map[bottom, right] + pts_map[top, left] - pts_map[bottom, left] - pts_map[top, right]
                sum_gray = integral_map[bottom, right] + integral_map[top, left] - integral_map[bottom, left] - integral_map[top, right]
                
                if pts_cnt > 0:
                    depth_map[i, j] = sum_gray / pts_cnt

        # 高斯模糊平滑
        s = wnd // 2 * 2 + 1
        s = min(s, 201)  # 最大窗口为201
        depth_map = cv2.GaussianBlur(depth_map, (s, s), s, s)

    return depth_map

def fill_disparity_map(disparity_map):
    """
    填充视差图中的空洞（视差图为16位类型）
    
    :param disparity_map: 输入的视差图（16位）
    :return: 填充后的视差图
    """
    # 将视差图转换为浮动类型并填充空洞
    disparity_map_float = disparity_map.astype(np.float32)

    # 填充视差图中的空洞
    filled_disparity = insert_depth_filled(disparity_map_float)
    
    # 将填充后的视差图转换回16位
    filled_disparity = np.clip(filled_disparity, 0, 255)  # 防止溢出
    return filled_disparity.astype(np.uint8)