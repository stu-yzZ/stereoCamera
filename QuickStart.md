# 快速开始

## 安装依赖

```bash
pip install opencv-python numpy matplotlib
```

## 运行项目

### 1. 查看3D点云（最快体验）

```bash
cd stereoCamera_py
python show3DCloud.py
```

### 2. 处理立体图像（核心功能）⭐

```bash
cd stereoCamera_py
python main_while.py
```

**✨ 重点说明**：代码中已内置相机的内外参数，可直接运行！

在 `main_while.py` 中，相机参数已配置：
```python
# 左相机内参
self.cam_matrix_l = np.array([
    [1733.74, 0.0, 792.27],
    [0.0, 1733.74, 541.89],
    [0., 0., 1.]
])

# 右相机内参
self.cam_matrix_r = np.array([
    [1733.74, 0.0, 792.27],
    [0.0, 1733.74, 541.89],
    [0., 0., 1.]
])

# 旋转矩阵和平移向量已设置
self.R = np.identity(3, dtype=np.float64)
self.T = np.array([[-536.62], [0.0], [0.0]])
```

**直接运行即可**：自动执行立体矫正 → 视差计算 → 深度图生成 → 点云输出

## 处理流程

```
左图 + 右图 → 立体矫正 → SGBM视差 → 深度图 → 3D点云
```

## 基本命令

```bash
cd stereoCamera_py

# 查看点云
python show3DCloud.py

# 处理图像对
python main_while.py

# 重新标定（可选）
python calibration_Stereo.py
```

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| 缺少依赖 | `pip install --upgrade opencv-python numpy matplotlib` |
| 点云为空 | 检查输入图像路径 |
| 内存不足 | 降低图像分辨率 |

详细文档请查看 [README.md](README.md)
