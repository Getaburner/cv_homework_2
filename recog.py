import cv2
import numpy as np
from config import CamaraMatrix
from utils import *
from config import *
import tracking
import plot

THRESH_ANGLE = 10
ENABLE_PREDICT = False
DETECT_COLOR_RED = False

cap = cv2.VideoCapture('sample.avi')

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	_, binary = cv2.threshold(gray, 54, 255, cv2.THRESH_BINARY)
	kernel = np.array((
		(0, 1, 0),
		(1, 1, 1),
		(0, 1, 0),
	), dtype='uint8')
	binary = cv2.erode(binary, kernel)
	cv2.imshow('bin', binary)
	contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = [cv2.approxPolyDP(c, 1, True)[:,0] for c in contours]
	#cv2.drawContours(img, contours,-1, (0,0,255), 2)

	found_contours = []
	colors = []
	directions = []
	for contour in contours:
		# 检测边数
		# 打印边数测试
		#cv2.putText(img, f'{len(contour)}', contour[0], cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
		#if len(contour) <= 1 or len(contour) >= 9:
		#	continue

		# 检测面积
		area = int(cv2.contourArea(contour))
		# 打印边数测试
		#cv2.putText(img, f'{area=}', contour[0], cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
		if area > 1000 or area < 10:
			continue

		# 检测颜色
		mask = np.zeros(binary.shape, np.uint8)
		cv2.drawContours(mask, (contour, ), -1, (255, ), -1)
		mean_bgr = cv2.mean(img, mask=mask)[:3]
		hsv = cv2.cvtColor(np.uint8(mean_bgr).reshape(1, 1, 3), cv2.COLOR_BGR2HSV)
		h = hsv[0, 0, 0]
		s = hsv[0, 0, 1]
		# 打印色调、饱和度测试
		#cv2.putText(img, f'{h=}', contour[0], cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
		#cv2.putText(img, f'{s=}', contour[0], cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
		isRed = 0 <= h <= 40
		isBlue = 80 <= h <= 110
		# 滤去色彩饱和度过低或不是红蓝两色的
		if s < 80 or not (isRed or isBlue):
			continue

		# 仅匹配预设的颜色
		if DETECT_COLOR_RED != isRed:
			continue

		# 检测方向
		#direction = getRadialDirection(contour)
		direction, _, _ = getMinRect(contour)

		# 打印朝向测试
		#cv2.putText(img, f'dir={int(direction)}', contour[0], cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)	

		found_contours.append(contour)
		colors.append(1 if isRed else 0)
		directions.append(direction)

	#cv2.drawContours(img, found_contours,-1, (0,255,0), 2)

	armorWidth = 135
	lightHeight = 55
	w = armorWidth / 2
	h = lightHeight / 2
	armor3d = np.array((
		(-w, -h, 0), # 左上
		(-w, h, 0), # 左下
		(w, -h, 0), # 右上
		(w, h, 0), # 右下
	), dtype='double')

	size = len(found_contours)
	cnt = 0
	plot.init_plot()
	found_points = []
	for i in range(size):
		for j in range(i+1, size):
			offset = abs(directions[i]-directions[j])
			if colors[i] == colors[j] and (offset < THRESH_ANGLE or offset > 180-THRESH_ANGLE):
				# 找到匹配的两个灯条
				line1 = getEndPoints(found_contours[i])
				line2 = getEndPoints(found_contours[j])

				if line1[0][0] + line1[1][0] < line2[0][0] + line2[1][0]:
					lnL, lnR = line1, line2
				else:
					lnL, lnR = line2, line1

				imagePoints = np.array((
					lnL[0],
					lnL[1],
					lnR[0],
					lnR[1],
				), dtype='double')

				# 解算得到三维位姿
				succ, _, offsetVec = cv2.solvePnP(armor3d, imagePoints, CamaraMatrix, DistortionCoefficients)
				offsetVec = offsetVec[:, 0]
				assert succ
				found_points.append(offsetVec)

				cv2.line(img, line1[0], line1[1], (0, 255, 255), 2)
				cv2.line(img, line2[0], line2[1], (0, 255, 255), 2)
				cv2.putText(img, chr(ord('A')+cnt), lnR[0], cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
				cnt += 1

	if ENABLE_PREDICT:
		tracking.update(found_points)
		dt, predicted_points = tracking.get_predict_points()
		cv2.putText(img, f'cost {int(dt*1000)}ms', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
		for p in predicted_points:
			plot.update_point(p, color='g')

	for p in found_points:
		plot.update_point(p, color='r')

	plot.commit()

	#cv2.drawContours(img, found_contours, -1, (255, 255, 255), 3)
	img = cv2.resize(img, (1280, 726))
	cv2.imshow('img', img)
	if cv2.waitKey(1) == ord('q'):
		break

