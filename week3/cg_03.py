import numpy as np
import cv2
import matplotlib.pyplot as plt

model_image = cv2.imread('models.png')
target_image = cv2.imread('target.jfif')
# HSV 변환
hsv = cv2.cvtColor(model_image, cv2.COLOR_BGR2HSV)
hsvt = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV)

# 오브젝트 히스토그램 추출 [H,S]
hist_roi = cv2.calcHist([hsv], [0, 1], None, [60, 64], [0, 180, 0, 256])

# 히스토그램 정규화 이후 대상 이미지에 역투영
cv2.normalize(hist_roi, hist_roi, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('histo', hist_roi)
cv2.waitKey(0)

dst = cv2.calcBackProject([hsvt], [0, 1], hist_roi, [0, 180, 0, 256], 1)
cv2.imshow('dst', dst)
cv2.waitKey(0)

# 타원모양 커널을 사용하여 컨벌루션
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cv2.filter2D(dst, -1, disc, dst)

# 역투영 결과를 이진화
ret, thresh = cv2.threshold(dst, 50, 255, 0)

# 이진화 이후 채널이 셋인 이미지로 바꾸고 대상 이미지와 and 연산
thresh = cv2.merge((thresh, thresh, thresh))
res = cv2.bitwise_and(target_image, thresh)

cv2.imshow('result', res)
cv2.waitKey(0)