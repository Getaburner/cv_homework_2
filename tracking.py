import numpy as np
import time
from utils import *
from ekf import CustomEKF

MAX_V = 6000
lastT = time.time()
points = [] # 0 = lastPos, 1 = curPos, 2 = EKF instance
dt = None

def update(newPosList: list):
    nplist = [p for p in newPosList]
    global points, lastT, dt
    curT = time.time()
    dt = curT - lastT
    lastT = curT
    maxD = MAX_V * dt
    deleting = []
    for i in range(len(points)):
        lastPos = points[i][0]
        index = None
        minDist = maxD
        for j, p in enumerate(nplist):
            curDist = pointDist(p, lastPos)
            if curDist < minDist:
                index = j
                minDist = curDist

        if index is not None:
            p = nplist.pop(index)
            points[i][1] = np.array(p)
        else:
            deleting.append(i)

    # 删除原先未被跟踪到的点
    for i in deleting[::-1]:
        points.pop(i)

    # 将未跟踪的点加入列表
    for p in nplist:
        assert len(p) == 3
        npinfo = [
                np.array((np.NaN, np.NaN, np.NaN)),
                p,
                CustomEKF(dt),
            ]
        points.append(npinfo)

    # 位置前后更替
    for i in range(len(points)):
        points[i][0] = points[i][1]


def get_predict_points():
    predicted = []
    for i in range(len(points)):
        # 跳过未开始预测的点
        if np.isnan(points[i][0][0]):
            continue

        ekfi = points[i][2]
        predict_point = ekfi.update_predict(points[i][1], dt)
        predicted.append(predict_point)

    return dt, predicted
