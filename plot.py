import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个绘图窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()

def init_plot():
	# 清除之前的点
	ax.clear()

# 用于实时更新的函数，这里假设 `get_new_point()` 函数用于获取实时的三维点位置
def update_point(point, color='r'):
	# 获取实时的三维点位置
	x = point[0]
	y = point[2]
	z = point[1]
	
	# 画出新的点
	ax.scatter(x, y, -z, color=color, marker='o')
	
	

def commit():
	# 设置坐标轴标签
	ax.set_xlabel('X')
	ax.set_zlabel('-Y')
	ax.set_ylabel('Z (depth)')

	ax.set_xlim(-1000, 1000)
	ax.set_zlim(-1000, 1000)
	ax.set_ylim(0, 10000)
	
	# 更新绘图
	plt.show()
	plt.pause(0.005)
