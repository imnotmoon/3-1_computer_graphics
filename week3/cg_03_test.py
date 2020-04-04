import numpy as np
import cv2
import matplotlib.pyplot as plt

quantizated_hue = 70        # 70
quantizated_sat = 100       # 100

model_image1 = cv2.imread('models.png')
model_image2 = cv2.imread('palette.png')
target_image = cv2.imread('target.jfif')
# HSV 변환
palette1 = cv2.cvtColor(model_image1, cv2.COLOR_BGR2HSV)
palette2 = cv2.cvtColor(model_image2, cv2.COLOR_BGR2HSV)
hsvt = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV)

# 오브젝트 히스토그램 추출 [H,S]
hist_temp1 = cv2.calcHist([palette1], [0, 1], None, [quantizated_hue, quantizated_sat], [0, 180, 0, 256])
hist_temp2 = cv2.calcHist([palette2], [0, 1], None, [quantizated_hue, quantizated_sat], [0, 180, 0, 256])
hist_roi = np.zeros((quantizated_hue, quantizated_sat))   # temp1, temp2를 합쳐서 여기 저장할 예정

# hist_for_plot = cv2.calcHist([palette1], [0, 1], None, [30, 32], [0, 180, 0, 256])

for i in range(quantizated_hue):
    for j in range(quantizated_sat) :
        hist_roi[i][j] = hist_temp1[i][j] + hist_temp2[i][j]

# cv2.imshow('hist', hist_roi)
# cv2.waitKey(0)


# 히스토그램 정규화 이후 대상 이미지에 역투영
hist_for_plot = cv2.normalize(hist_roi, hist_roi, 0, 255, cv2.NORM_MINMAX)  # 출력용 정규화된 히스토그램
dst = cv2.calcBackProject([hsvt], [0, 1], hist_roi, [0, 180, 0, 256], 1)

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