import cv2
import matplotlib.pyplot as plt
import numpy as np
from msvcrt import getch

_QUANTIZATED_HUE_ = 90  # 색조 : 70
_QUANTIZATED_SATURATION_ = 128  # 채도 : 100
_VAL_THRESHOLD_ = 30  # 임계값 : 50
gaussian_kernel_size = 5  # 가우시안 필터 사이즈 : 무조건 홀수여야 함.


# 스페이스바 체크 함수
def space_check():
    while True:
        if ord(getch()) == 32:
            break


# trackbar 수치 변동시 함수를 호출하는데, 딱히 필요한 기능이 없으므로 빈 함수 호출
def nothing(x):
    pass


def detect_face(input_frame, val_threshold):
    # 가우시안 블러
    blurred = cv2.GaussianBlur(input_frame, (gaussian_kernel_size, gaussian_kernel_size), 0)

    # 입력영상 HSV 변환
    hsvt = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 모델 히스토그램을 입력 영상에 역투영
    dst = cv2.calcBackProject([hsvt], [0, 1], hist_roi, [0, 180, 0, 256], 1)

    # 타원모양 커널을 사용하여 컨벌루션 : 신뢰도 값이 높은 픽셀 주변으로 타원을 그리고 그 범위까지 인정
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cv2.filter2D(dst, -1, disc, dst)

    # 임계값을 이용하여 이진화
    ret, thresh = cv2.threshold(dst, val_threshold, 255, cv2.THRESH_BINARY)

    # 이진화 이후 채널이 셋인 이미지로 바꾸고 대상 이미지와 and 연산
    thresh_with_critical_value = cv2.merge((thresh, thresh, thresh))
    res_with_critical_value = cv2.bitwise_and(blurred, thresh_with_critical_value)
    return res_with_critical_value


# 모델이미지 로드
model_image = cv2.imread("palette.png")
# HSV 변환
palette = cv2.cvtColor(model_image, cv2.COLOR_BGR2HSV)
# 모델 히스토그램 추출
hist_roi = cv2.calcHist([palette], [0, 1], None, [_QUANTIZATED_HUE_, _QUANTIZATED_SATURATION_], [0, 180, 0, 256])
# 모델 히스토그램 정규화
cv2.normalize(hist_roi, hist_roi, 0, 255, cv2.NORM_MINMAX)

print("캠을 켜려면 스페이스바를 누르십시오.\n")
space_check()

# 캠으로부터 영상입력을 받고 영상의 크기를 설정함
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows
# cap = cv2.VideoCapture(0)                       # Mac
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if cap.isOpened():
    # ret : 카메라 사용 성공 여부 => True / False
    ret, frame = cap.read()
else:
    ret = False
    print("카메라를 사용할 수 없습니다!\n")
    exit()

# 출력할 이미지 창에 trackbar 설정 (기본값 50, 최소 1 최대 255)
cv2.namedWindow("FACE_res")
cv2.createTrackbar("_VAL_THRESHOLD", "FACE_res", 50, 255, nothing)
cv2.setTrackbarMin("_VAL_THRESHOLD", "FACE_res", 1)
print("원본 영상과 얼굴 검출 결과 출력\n\n창을 종료하려면 출력창에서 스페이스바를 누르십시오.\n")

while ret:  # 카메라가 영상을 받아오면 True, 못받으면 break
    ret, frame = cap.read()

    # trackbar 수치에 따라 임계값 변동
    _VAL_THRESHOLD_ = cv2.getTrackbarPos("_VAL_THRESHOLD", "FACE_res")
    result = detect_face(frame, _VAL_THRESHOLD_)

    # 원본 이미지와 검출 결과 이미지를 가로로 합친뒤 출력
    res = cv2.hconcat([frame, result])
    cv2.imshow("FACE_res", res)

    # escape key : space bar 누르면 종료
    if cv2.waitKey(1) == 32:
        break

# close window : windowName("Live Video Feed")
cv2.destroyAllWindows()
# release camera
cap.release()
