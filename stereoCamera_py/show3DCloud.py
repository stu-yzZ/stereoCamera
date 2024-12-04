import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D  #输出三维点必须要导入此包，否则会 Error：3d
import matplotlib
matplotlib.use('TkAgg')

x, y, z, r, g, b = np.loadtxt('point_cloud.xyz').T  #读取点云xyz文件
# print(x,y,z)                      #测试一下，看看能不能输出我们想要的x，y，z
 
fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = fig.add_subplot(projection='3d')
# 设置坐标轴的范围
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))
ax.set_zlim(np.min(z), np.max(z))

# 将颜色归一化到[0, 1]范围
colors = np.vstack((r, g, b)).T / 255.0  # 假设r, g, b的范围是0-255，归一化到[0, 1]

# 在3D空间中绘制点云，传入颜色参数
ax.scatter(x, y, z, c=colors, s=10, marker='.', depthshade=True)

# 隐藏坐标轴
ax.axis('off')

# 设置初始视角
ax.view_init(azim=90, elev=90)

# 显示图形
plt.show()

