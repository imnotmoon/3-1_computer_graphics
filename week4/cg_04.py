import cv2
import matplotlib.pyplot as plt
import numpy as np

_QUANTIZATED_HUE_ = 100              # 색조 : 70
_QUANTIZATED_SATURATION_ = 150      # 채도 : 100
_VAL_THRESHOLD_ = 70                # 임계값 : 50

# 모델이미지 로드 + HSV변환 + 히스토그램 추출
model_image = cv2.imread("models.png")
palette = cv2.cvtColor(model_image, cv2.COLOR_BGR2HSV)
hist_roi = cv2.calcHist([palette], [0, 1], None, [_QUANTIZATED_HUE_, _QUANTIZATED_SATURATION_], [0, 180, 0, 256])

def detect_face(frame) :
    dst = cv2.calcBackProject([frame], [0, 1], hist_roi, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, _VAL_THRESHOLD_, 255, cv2.THRESH_BINARY)
    thresh_with_critical_value = cv2.merge((thresh, thresh, thresh))
    res_with_critical_value = cv2.bitwise_and(frame, thresh_with_critical_value)

    return res_with_critical_value

# 캠
cap = cv2.VideoCapture(0)
print("camera initialized")

gaussian_kernel_size = 5

while True:
    if cap.isOpened() :
        # ret : 카메라 사용 성공 여부 => True / False
        ret, frame = cap.read()
    else :
        ret = False

    while ret:      # 카메라가 영상을 받아오면 True, 못받으면 break
        ret, frame = cap.read()

        # 가우시안 블러
        dst = cv2.GaussianBlur(frame, (5,5), 0)

        # 흑백으로 영상 변환
        output = detect_face(dst)

        # 화면 출력
        cv2.imshow("FACE", output)

        # escape key : esc 누르면 종료
        if cv2.waitKey(1) == 27:
            break

    # close window : windowName("Live Video Feed")
    cv2.destroyWindow("FACE")
    # release camera
    cap.release()

