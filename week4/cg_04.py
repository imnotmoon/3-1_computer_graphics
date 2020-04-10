import cv2
import matplotlib.pyplot as plt
import numpy as np

_QUANTIZATED_HUE_ = 100  # 색조 : 70
_QUANTIZATED_SATURATION_ = 150  # 채도 : 100
_VAL_THRESHOLD_ = 30  # 임계값 : 50


# trackbar 수치 변동시 함수를 호출하는데, 딱히 필요한 기능이 없으므로 빈 함수 호출
def nothing(x):
    pass


def detect_face(frame, val_threshold):
    # 가우시안 블러
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # 입력영상 HSV 변환
    hsvt = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 모델 히스토그램을 입력 영상에 역투영
    dst = cv2.calcBackProject([hsvt], [0, 1], hist_roi, [0, 180, 0, 256], 1)

    # 타원모양 커널을 사용하여 컨벌루션 : 신뢰도 값이 높은 픽셀 주변으로 타원을 그리고 그 범위까지 인정
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)

    # 임계값을 이용하여 이진화
    ret, thresh = cv2.threshold(dst, val_threshold, 255, cv2.THRESH_BINARY)

    # 이진화 이후 채널이 셋인 이미지로 바꾸고 대상 이미지와 and 연산
    thresh_with_critical_value = cv2.merge((thresh, thresh, thresh))
    res_with_critical_value = cv2.bitwise_and(blurred, thresh_with_critical_value)
    return res_with_critical_value


# 모델이미지 로드
model_image = cv2.imread("models.png")
# HSV 변환
palette = cv2.cvtColor(model_image, cv2.COLOR_BGR2HSV)
# 모델 히스토그램 추출
hist_roi = cv2.calcHist([palette], [0, 1], None, [_QUANTIZATED_HUE_, _QUANTIZATED_SATURATION_], [0, 180, 0, 256])
# 모델 히스토그램 정규화
cv2.normalize(hist_roi, hist_roi, 0, 255, cv2.NORM_MINMAX)

# 캠
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("camera initialized")
gaussian_kernel_size = 5

# 출력할 이미지 창에 trackbar 설정 (기본값 50, 최소 1 최대 255)
cv2.namedWindow("FACE_res")
cv2.createTrackbar("_VAL_THRESHOLD", "FACE_res", 50, 255, nothing)
cv2.setTrackbarMin("_VAL_THRESHOLD", "FACE_res", 1)

while True:
    if cap.isOpened():
        # ret : 카메라 사용 성공 여부 => True / False
        ret, frame = cap.read()
    else:
        ret = False

    while ret:  # 카메라가 영상을 받아오면 True, 못받으면 break
        ret, frame = cap.read()

        # trackbar 수치에 따라 임계값 변동
        _VAL_THRESHOLD_ = cv2.getTrackbarPos("_VAL_THRESHOLD", "FACE_res")
        result=detect_face(frame,_VAL_THRESHOLD_)
        # 화면 출력
        cv2.imshow("FACE_ori", frame)
        cv2.imshow("FACE_res", result)

        # escape key : esc 누르면 종료
        if cv2.waitKey(1) == 27:
            break

    # close window : windowName("Live Video Feed")
    cv2.destroyAllWindows()
    # release camera
    cap.release()
    exit()
