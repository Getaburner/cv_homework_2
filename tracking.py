import numpy as np
import time
from utils import *

MAX_V = 5000
lastT = time.time()
# [i][0][0:3]curPos, [i][0][3:6]lastPos, [i][0][6:9]undefined
# [i][1][0:3]curV, [i][1][3:6]lastV, [i][1][6:9]curA
# [i][2][0:9]KF_x
# [i][3][0:9]KF_p
points = np.zeros((0, 4, 9))
dt = None

def update(newPosList: list):
    nplist = [p for p in newPosList]
    global points, lastT, dt
    curT = time.time()
    dt = curT - lastT
    lastT = curT

    maxD = MAX_V * dt
    deleting = []
    for i in range(points.shape[0]):
        lastPos = points[i, 0, 3:6]
        index = None
        minDist = maxD
        for j, p in enumerate(nplist):
            curDist = pointDist(p, lastPos)
            if curDist < minDist:
                index = j
                minDist = curDist

        if index is not None:
            p = nplist.pop(index)
            points[i, 0, 0:3] = np.array(p)
        else:
            deleting.append(i)

    # 删除原先未被跟踪到的点
    points = np.delete(points, deleting, axis=0)

    # 将未跟踪的点加入列表
    for p in nplist:
        assert len(p) == 3
        # 同时将卡尔曼滤波参数初始化
        newpoint = np.array([
            [*p, *[np.NaN]*6],
            [np.NaN]*9,
            [0]*9,
            [1]*9,
        ]).reshape(4, 9)
        points = np.append(points, (newpoint, ), axis=0)

    # 速度量前后更替
    points[:, 1, 3:6] = points[:, 1, 0:3]
    for i in range(points.shape[0]):
        lastPos = points[i, 0, 3:6]
        if not np.isnan(lastPos[0]):
            # 计算速度
            points[i, 1, 0:3] = (points[i, 0, 0:3] - lastPos) / dt

        # 更新加速度
        lastV = points[i, 1, 3:6]
        if not np.isnan(lastV[0]):
            points[i, 1, 6:9] = (points[i, 1, 0:3] - lastV) / dt

    # 位置前后更替
    points[:, 0, 3:6] = points[:, 0, 0:3]

    # 状态转移矩阵
    da = 0.5*dt**2
    F = np.array([[1, 0, 0, dt,0 ,0 ,da, 0,  0],
                  [0, 1, 0, 0, dt,0 ,0 , da, 0],
                  [0, 0, 1, 0, 0, dt,0 , 0,  da],
                  [0, 0, 0, 1, 0, 0, dt, 0,  0],
                  [0, 0, 0, 0, 1, 0, 0,  dt, 0],
                  [0, 0, 0, 0, 0, 1, 0,  0,  dt],
                  [0, 0, 0, 0, 0, 0, 1,  0,  0],
                  [0, 0, 0, 0, 0, 0, 0,  1,  0],
                  [0, 0, 0, 0, 0, 0, 0,  0,  1]])

    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],     # 观测矩阵
                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0]])

    Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 过程噪声
    R = np.diag([0.1, 0.1, 0.1])  # 测量误差方差

    
    for i in range(points.shape[0]):
        measure = np.concatenate((points[i, 0, 0:3], points[i, 1, 0:3], points[i, 1, 6:9]))
        if np.any(np.isnan(measure)):
            continue

        x = points[i, 2].T
        P = points[i, 3]  # 状态估计方差

        # 预测
        x = np.dot(F, x) # 状态转移后
        P = np.dot(F, np.dot(P, F.T)) + Q 
        
        # 更新
        y = measure.reshape(-1, 1) - x
        S = np.dot(H, np.dot(P, H.T)) + R
        K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))
        
        print(K.shape, y.shape)
        x = x + np.dot(K, y)
        P = P - np.dot(K, np.dot(H, P))
        points[i, 2] = x.T
        points[i, 3] = P

def get_predict_points():
    predicted = []
    for i in range(points.shape[0]):
        # 跳过未开始预测的点
        if np.isnan(points[i, 1, 6]):
            continue
        predicted.append(points[i, 2, 0:3])

    return dt, predicted
