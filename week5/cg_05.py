import cv2
import matplotlib.pyplot as plt
import numpy as np
from msvcrt import getch


# 스페이스바 체크 함수
def space_check():
    while True:
        if ord(getch()) == 32:
            break


# trackbar 수치 변동시 함수를 호출하는데, 딱히 필요한 기능이 없으므로 빈 함수 호출
def nothing(x):
    pass


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

# 출력할 이미지 창에 최소임계값, 최대임계값의 trackbar 설정 (기본값 50, 최소 1 최대 1000)
cv2.namedWindow("result")
cv2.createTrackbar("_VAL_THRESHOLD_MIN", "result", 50, 1000, nothing)
cv2.createTrackbar("_VAL_THRESHOLD_MAX", "result", 50, 1000, nothing)
cv2.setTrackbarMin("_VAL_THRESHOLD_MIN", "result", 1)
cv2.setTrackbarMin("_VAL_THRESHOLD_MAX", "result", 1)

print("원본 영상과 캐니 에지 검출 결과 출력\n\n창을 종료하려면 출력창에서 스페이스바를 누르십시오.\n")

while ret:  # 카메라가 영상을 받아오면 True, 못받으면 break
    ret, frame = cap.read()

    # trackbar 수치에 따라 임계값 변동
    _VAL_THRESHOLD_MIN = cv2.getTrackbarPos("_VAL_THRESHOLD_MIN", "result")
    _VAL_THRESHOLD_MAX = cv2.getTrackbarPos("_VAL_THRESHOLD_MAX", "result")

    canny = cv2.Canny(frame, _VAL_THRESHOLD_MIN, _VAL_THRESHOLD_MAX)
    # 이진화된 이미지를 채널이 셋인 이미지로 변환
    canny = cv2.merge((canny, canny, canny))

    # 원본 이미지와 검출 결과 이미지를 가로로 합친뒤 출력
    res = cv2.hconcat([frame, canny])
    cv2.imshow("result", res)

    # escape key : space bar 누르면 종료
    if cv2.waitKey(1) == 32:
        break

# close window : windowName("Live Video Feed")
cv2.destroyAllWindows()
# release camera
cap.release()
