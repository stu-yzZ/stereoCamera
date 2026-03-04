# 双目立体相机（Stereo Camera）

## 项目概述

这是一个完整的双目立体相机处理系统，涵盖相机标定、视差计算、深度重建和点云生成等关键功能。通过左右相机配对图像，实现三维场景重建。

**核心技术栈**：OpenCV、NumPy、立体匹配（SGBM）、点云处理

---

## 项目特性

- ✅ **双目相机标定**：基于棋盘格的摄像头内外参数标定
- ✅ **图像矫正**：立体图像对的极线矫正处理
- ✅ **视差计算**：采用SGBM算法进行实时视差图生成
- ✅ **深度图生成**：从视差图转换为深度图，含孔洞填充算法
- ✅ **点云重建**：从深度图生成三维点云数据
- ✅ **点云可视化**：使用matplotlib进行3D点云显示

---

## 项目结构

```
stereoCamera/
├── README.md                          # 项目文档
├── stereoCamera_py/                   # Python实现主目录
│   ├── calibration_Stereo.py          # 相机标定脚本
│   ├── main_if.py                     # 主程序（if版本）
│   ├── main_while.py                  # 主程序（while版本）
│   ├── show3DCloud.py                 # 点云三维可视化脚本
│   ├── solution.py                    # 辅助函数库（深度填充等）
│   ├── point_cloud.xyz                # 点云数据文件1
│   ├── point_cloud1.xyz               # 点云数据文件2
│   ├── readme.txt                     # 说明文档
│   └── imagefolder/                   # 图像数据文件夹
│       └── 数据集地址.txt              # 数据集说明
```

---

## 核心模块说明

### 1. **calibration_Stereo.py** - 相机标定
- 功能：对双目相机系统进行标定
- 输入：左右摄像头的棋盘格标定图像对
- 输出：相机内参矩阵、畸变系数、旋转矩阵和平移矩阵
- 棋盘格配置：8×5（列×行）

**核心参数**：
- 相机内参矩阵 (Camera Matrix)：3×3矩阵
- 畸变系数 (Distortion Coefficients)：[k1, k2, p1, p2, k3]
- 旋转矩阵 (Rotation Matrix)：R
- 平移向量 (Translation Vector)：T

### 2. **main_if.py / main_while.py** - 主处理流程
完整的立体视觉处理管线：

1. **图像加载与预处理**
   - 读取左右相机图像对
   - 转换为灰度图像

2. **立体矫正** (Rectification)
   - 计算矫正变换矩阵
   - 消除极线失配问题

3. **视差计算** (Disparity Estimation)
   - 使用OpenCV的SGBM (Semi-Global Block Matching) 算法
   - 参数可调：块大小、视差范围等

4. **视差图处理**
   - 视差图数据类型转换
   - 孔洞填充和滤波处理

5. **深度图生成**
   - 根据公式：Depth = (baseline × focal_length) / disparity
   - 深度单位通常为米

6. **点云生成**
   - 从深度图重投影为3D坐标
   - 包含RGB颜色信息

### 3. **solution.py** - 辅助算法
提供深度图和视差图的孔洞填充函数：

```python
insert_depth_filled(depth_map)      # 填充深度图中的孔洞
fill_disparity_map(disparity_map)   # 填充视差图中的孔洞
```

**填充策略**：
- 使用积分图加速计算
- 多尺度窗口逐步填充
- 高斯模糊平滑处理

### 4. **show3DCloud.py** - 点云可视化
- 读取点云XYZ文件格式
- 三维散点图展示
- 支持RGB颜色映射
- 交互式视角控制（旋转、平移、缩放）

---

## 使用方法

### 前置要求

```bash
pip install opencv-python numpy matplotlib
```

### 1. 相机标定

准备标定图像（左右摄像头的棋盘格图像对），放置在 `stereoImage/` 文件夹下：
- 左图文件名：`*_L.jpg`
- 右图文件名：`*_R.jpg`

运行标定脚本：
```bash
python calibration_Stereo.py
```

输出：`camera_calibration_data.npz` 文件（包含标定参数）

### 2. 立体视觉处理

使用标定参数处理立体图像对：

```bash
# 两个版本的主程序，逻辑相同，实现略有差异
python main_if.py      # 使用if-else分支版本
python main_while.py   # 使用while循环版本
```

**处理流程**：
1. 加载左右图像
2. 立体矫正
3. 计算视差
4. 生成深度图
5. 输出点云数据

### 3. 点云可视化

```bash
python show3DCloud.py
```

- 自动加载 `point_cloud.xyz` 文件
- 显示三维点云
- 可调整视角：鼠标旋转/缩放

---

## 关键参数配置

### stereoCamera 类参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `cam_matrix_l` | 左相机内参矩阵 | 3×3 矩阵 |
| `cam_matrix_r` | 右相机内参矩阵 | 3×3 矩阵 |
| `distortion_l` | 左相机畸变系数 | [k1, k2, p1, p2, k3] |
| `distortion_r` | 右相机畸变系数 | [k1, k2, p1, p2, k3] |
| `R` | 旋转矩阵 | 3×3 旋转矩阵 |
| `T` | 平移向量 | 基线距离等 |
| `baseline` | 双目基线长度 | 536.62 像素 |

### SGBM 算法参数

- `minDisparity`：最小视差值
- `numDisparities`：视差搜索范围
- `blockSize`：匹配块大小
- `speckleWindowSize`：斑点滤波窗口
- `speckleRange`：斑点滤波范围

---

## 点云文件格式

`.xyz` 文件格式（纯文本，空格分隔）：
```
x1 y1 z1 r1 g1 b1
x2 y2 z2 r2 g2 b2
...
```

- **x, y, z**：三维坐标
- **r, g, b**：RGB颜色值（0-255）

---

## 主要函数说明

### stereoCamera.getRectifyTransform()
计算立体矫正的变换矩阵

### stereoCamera.getDisparityMap()
计算立体视差图

### stereoCamera.getDepthMap()
从视差图生成深度图

### stereoCamera.getPointCloud()
从深度图生成三维点云

---

## 文件输出

- `camera_calibration_data.npz`：标定参数
- `disparity_map.png`：视差图可视化
- `depth_map.png`：深度图可视化
- `point_cloud.xyz`：三维点云数据

---

## 常见问题

### Q: 标定失败提示找不到棋盘角点？
A: 检查棋盘格图像质量和光照条件，确保棋盘格清晰可识别

### Q: 点云中有大量噪点？
A: 调整SGBM参数的 `speckleRange` 和滤波窗口大小

### Q: 深度图中有孔洞怎么办？
A: 使用 `solution.py` 中的 `insert_depth_filled()` 函数进行填充

### Q: 内存占用过高？
A: 减小图像分辨率或调整SGBM的 `numDisparities` 参数

---

## 技术原理

### 双目视觉基本原理
```
深度 D = (基线 B × 焦距 f) / 视差 d
```

### 极线矫正
- 通过旋转和平移，使对应点位于同一水平扫描线上
- 加快匹配速度并提高精度

### SGBM 算法
- Semi-Global Block Matching
- 结合局部块匹配和全局优化
- 提供较好的精度和效率平衡

