import cv2
import numpy as np

def cos(v1, v2):
	return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def pointDist(p1, p2):
	return np.linalg.norm(p1-p2)

def deg2rad(deg):
	return deg / 180 * np.pi

def rad2deg(rad):
	return rad / np.pi * 180

def getAngle(v):
	haxis = np.array((1, 0))
	origAngle = np.arccos(cos(v, haxis))
	if v[1] < 0:
		return np.pi * 2 - origAngle
	else:
		return origAngle

def getCenter(contour):
	x, y, w, h = cv2.boundingRect(contour)
	return np.array((x+w/2, y+h/2), dtype=int)

def getRadialDirection(contour):
	rad = 0
	max_dist = 0
	for i in range(len(contour)):
		for j in range(i+1, len(contour)):
			v = (contour[i] - contour[j])
			dist = np.linalg.norm(v)
			if dist < max_dist:
				continue

			max_dist = dist
			rad = getAngle(v)

	deg = rad2deg(rad)
	if deg > 180:
		deg -= 180
	return deg

def getMinRect(contour):
	(cx, cy), (w, h), theta = cv2.minAreaRect(contour)
	assert theta >= 0
	if w < h:
		deg = 90 - theta
		ls, ss = h, w
	else:
		deg = 180 - theta
		ls, ss = w, h

	return deg, (cx, cy), (ls, ss)

def drawRect(contour):
	margin = 10
	x, y, w, h = cv2.boundingRect(contour)
	cv2.rectangle(img, (x-margin, y-margin), (x+w+margin, y+h+margin), (0, 255, 0), 3)

def getEndPoints(contour, dtype=int):
	angle, (cx, cy), size = getMinRect(contour)
	rad = deg2rad(angle)
	r = size[0] / 2
	dx = r * np.cos(rad)
	dy = r * np.sin(rad)
	p1 = (dtype(cx + dx), dtype(cy - dy))
	p2 = (dtype(cx - dx), dtype(cy + dy))

	# 靠上的点在前
	return (p1, p2)
