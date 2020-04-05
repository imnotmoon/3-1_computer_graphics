import numpy as np
import cv2
import matplotlib.pyplot as plt
# import sys, tty, termios        # 맥에서 엔터 없이 인풋받아오는게 안돼서..

########### 최적화는 이거 수정하면 될것 같습니다 #############
_QUANTIZATED_HUE_ = 180              # 색조 : 70
_QUANTIZATED_SATURATION_ = 256      # 채도 : 100
_VAL_THRESHOLD_ = 50                # 임계값 : 50

# 스페이스바 입력시 plt창을 끄게 만드는 함수입니다.
def quit_figure(event):
    if event.key == ' ':
        plt.close(event.canvas.figure)


# 이미지 로드
print("이미지 로드\n")
model_image1 = cv2.imread('models.png')
# model_image2 = cv2.imread('palette.png')
target_image = cv2.imread('target.jfif')

# HSV 변환
print("HSV 변환\n")
palette1 = cv2.cvtColor(model_image1, cv2.COLOR_BGR2HSV)
# palette2 = cv2.cvtColor(model_image2, cv2.COLOR_BGR2HSV)
hsvt = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV)

# 오브젝트 히스토그램 추출 [H,S]
# 2-A
print("오브젝트 히스토그램 추출 [H,S]\n")

# 모든 픽셀을 탐색하여 히스토그램 생성하는 for문
# height, width = palette1.shape[:2]
# hist_roi=np.zeros((_QUANTIZATED_HUE_,_QUANTIZATED_SATURATION_),)
# a=180/_QUANTIZATED_HUE_
# b=256/_QUANTIZATED_SATURATION_
#
# for j in range(height):
#     for i in range(width):
#         h=int(palette1[j][i][0]/a)
#         s=int(palette1[j][i][1]/b)
#         hist_roi[h][s]+=1

hist_roi = cv2.calcHist([palette1], [0, 1], None, [_QUANTIZATED_HUE_, _QUANTIZATED_SATURATION_], [0, 180, 0, 256])
# hist_temp2 = cv2.calcHist([palette2], [0, 1], None, [_QUANTIZATED_HUE_, _QUANTIZATED_SATURATION_], [0, 180, 0, 256])
# hist_roi = np.zeros((_QUANTIZATED_HUE_, _QUANTIZATED_SATURATION_))   # temp1, temp2를 합쳐서 여기 저장할 예정
# print(np.shape(hist_temp1)[0], np.shape(hist_temp1)[1])


# for i in range(_QUANTIZATED_HUE_):
#     for j in range(_QUANTIZATED_SATURATION_) :
#         hist_roi[i][j] = hist_temp1[i][j] + hist_temp2[i][j]

# 2-C
print("정규화한 히스토그램의 그래프\n")
cv2.normalize(hist_roi, hist_roi, 0, 255, cv2.NORM_MINMAX)  # 출력용 정규화된 히스토그램 : HUE
cv2.imshow('hist', hist_roi)
cv2.waitKey(0)


# 히스토그램 정규화 이후 대상 이미지에 역투영
print("대상 이미지에 히스토그램 역투영\n")
dst = cv2.calcBackProject([hsvt], [0, 1], hist_roi, [0, 180, 0, 256], 1)

# 찾은 픽셀들 : 신뢰도 맵
# 3-A
cv2.imshow('dst', dst)
cv2.waitKey(0)

# 4-A       # cv2로 그래프 찍으니까 잘 안보임, range를 [0, 256]으로 하면 전부다 검은색(배경)이라 잘 안보여서 [1, 256]으로 함.
print("신뢰도 맵 히스토그램의 그래프\n")
dst_hist = cv2.calcHist([dst], [0], None, [256], [1, 256])
plt.plot(dst_hist)
cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
plt.show()

# 타원모양 커널을 사용하여 컨벌루션 : 신뢰도 값이 높은 픽셀 주변으로 타원을 그리고 그 범위까지 인정
print("타원모양 커널을 사용하여 컨벌루션\n")
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cv2.filter2D(dst, -1, disc, dst)

# # 찾은 픽셀들을 컨벌루션한거
cv2.imshow('dst', dst)
cv2.waitKey(0)


# 역투영 결과를 임계값을 이용해 이진화(threshold)
# 4-C
ret, thresh = cv2.threshold(dst, _VAL_THRESHOLD_, 255, cv2.THRESH_BINARY)
# 이진화 이후 채널이 셋인 이미지로 바꾸고 대상 이미지와 and 연산
thresh_with_critical_value = cv2.merge((thresh, thresh, thresh))
res_with_critical_value = cv2.bitwise_and(target_image, thresh_with_critical_value)
print("얼굴 검출 최종결과\n")
cv2.imshow('result', res_with_critical_value)
cv2.waitKey(0)


# 오츄의 알고리즘을 이용한 이진화
ret2, thresh2 = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
thresh_with_otsu_algorithm = cv2.merge((thresh2, thresh2, thresh2))
res_with_otsu_algorithm = cv2.bitwise_and(target_image, thresh_with_otsu_algorithm)

# 5-A 오츄의 알고리즘이 정한 임계값
print("오츄의 알고리즘 임계값: ", ret2)

# 6-C 오츄 알고리즘를 이용한 얼굴 검출 결과
print("오츄의 알고리즘을 이용한 최종결과\n")
cv2.imshow('resultByOtsu', res_with_otsu_algorithm)
cv2.waitKey(0)