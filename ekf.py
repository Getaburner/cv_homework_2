import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

def fx(x, dt):
	# 更新状态转移矩阵以包括加速度
	F = np.array([[1, dt, 0.5*dt**2, 0, 0, 0, 0, 0, 0],
				  [0, 1, dt, 0, 0, 0, 0, 0, 0],
				  [0, 0, 1, 0, 0, 0, 0, 0, 0],
				  [0, 0, 0, 1, dt, 0.5*dt**2, 0, 0, 0],
				  [0, 0, 0, 0, 1, dt, 0, 0, 0],
				  [0, 0, 0, 0, 0, 1, 0, 0, 0],
				  [0, 0, 0, 0, 0, 0, 1, dt, 0.5*dt**2],
				  [0, 0, 0, 0, 0, 0, 0, 1, dt],
				  [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)
	return np.dot(F, x)

def hx(x):
	# 实际测量的位置
	return np.array([x[0], x[3], x[6]])

def jacobian_F(x, dt):
	# 状态转移函数的雅可比矩阵，包括加速度
	# 没懂
	return np.array([[1, dt, 0.5*dt**2, 0, 0, 0, 0, 0, 0],
					 [0, 1, dt, 0, 0, 0, 0, 0, 0],
					 [0, 0, 1, 0, 0, 0, 0, 0, 0],
					 [0, 0, 0, 1, dt, 0.5*dt**2, 0, 0, 0],
					 [0, 0, 0, 0, 1, dt, 0, 0, 0],
					 [0, 0, 0, 0, 0, 1, 0, 0, 0],
					 [0, 0, 0, 0, 0, 0, 1, dt, 0.5*dt**2],
					 [0, 0, 0, 0, 0, 0, 0, 1, dt],
					 [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)

def jacobian_H(x):
	# 测量函数的雅可比矩阵
	# 同上
	return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
					 [0, 0, 0, 1, 0, 0, 0, 0, 0],
					 [0, 0, 0, 0, 0, 0, 1, 0, 0]])


class CustomEKF(EKF):
	def __init__(self, dt) -> None:
		super().__init__(dim_x=9, dim_z=3)
		self.x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
		self.P *= 100  # 初始状态协方差
		self.R = np.eye(3) * 0.6  # 测量噪声
		self.Q = block_diag(Q_discrete_white_noise(3, dt, 0.1),  # x方向的过程噪声
						   Q_discrete_white_noise(3, dt, 0.1),  # y方向的过程噪声
						   Q_discrete_white_noise(3, dt, 0.1))  # z方向的过程噪声

	def update_predict(self, z, dt: float):
		self.predict()
		self.update(z, HJacobian=jacobian_H, Hx=hx)
		return self.x[:3]
