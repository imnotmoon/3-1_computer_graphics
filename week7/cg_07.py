import cv2
import numpy as np

# opencv 스페이스바 체크 함수
def cv_space_check():
    while True:
        if cv2.waitKey(1) == 32: break


################################
# 1. 번호판 영상을 준비한다.
#   ==> 다른 입력영상(input2)도 정상적으로 작동하는것 확인했습니다.

# 2-B 영상의 크기는 번호판 보다 가로 세로가 10% 더 크게 되도록 조정할 것
#   ==> 무슨 의미인지 모르겠습니다.

# 3-A 적절한 thresholding 기법 사용하기
#   ==> adaptive thresholding 사용했습니다. (방식은 가우시안, 블록사이즈와 C 파라미터 최적화 필요)

# 4-A 이진 모폴로지 연산을 이용하여 잡음제거 및 윤곽 스무싱
#   ==> SE는 5*5 직사각형(MORPH_RECT) 사용했고, 연산은 열기->닫기 진행했습니다.
################################


model_image = cv2.imread('input2.jpg')
print("이미지 읽기 완료")

# 그림의 꼭짓점 좌표를 저장할 변수 (순서 : 왼쪽 위 - 오른쪽 위 - 왼쪽 아래 - 오른쪽 아래)
src = []

# 실제 자동차 번호판의 비율 측정 후 만든 적절한 크기의 template
dst_np = np.array([[0, 0], [400, 0], [0, 100], [400, 100]], dtype=np.float32)


# 마우스로 이미지에 점을 4개 찍고 점의 좌표를 src에 저장
def mouse_handler(event, x, y, flags, param):
    # 콜백
    if event == cv2.EVENT_LBUTTONUP:
        img = model_image.copy()
        # 클릭지점 저장
        src.append([x, y])
        for xx, yy in src:
            # 클릭한 위치에 빨간원을 그림
            cv2.circle(img, center=(xx, yy), radius=5, color=(0, 0, 255), thickness=-1)

        cv2.imshow('img', img)

        if len(src) == 1: print(src[0], "\n오른쪽 위 점을 클릭해 주세요.")
        if len(src) == 2: print(src[1], "\n왼쪽 밑 점을 클릭해 주세요.")
        if len(src) == 3: print(src[2], "\n오른쪽 밑 점을 클릭해 주세요.")
        if len(src) == 4: print(src[3], "\n변환행렬 생성을 위해 이미지창에서 스페이스바를 눌러주세요.\n")


cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse_handler)
cv2.imshow('img', model_image)

print("\n왼쪽 위 점을 클릭해 주세요.")

while True:
    if cv2.waitKey(1) == 32:  # 스페이스바가 입력되고
        if len(src) == 4:  # 4개의 점이 찍혔을 경우 진행
            break


# 점의 좌표를 float32 np array 타입으로 저장
src_np = np.array(src, dtype=np.float32)

# 변환행렬 생성
mat = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)

print("변환행렬\n", mat, "\n결과영상 출력을 위해 이미지창에서 스페이스바를 눌러주세요.\n")
cv_space_check()


# 변환영상 출력
dst = cv2.warpPerspective(model_image, M=mat, dsize=(400, 100))
cv2.imshow('res', dst)

print("결과 영상 출력 완료\n진행하기 위해 이미지창에서 스페이스바를 눌러주세요.\n")
cv_space_check()


# 3-A 적응형 임계처리 함수의 파라미터 (최적화 필요 3-B)
blockSize = 21
C = 3

# 번호판 이미지를 그레이스케일 이미지로 변환후, 적응형 임계처리 적용
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
thres = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)
cv2.imshow('res2', thres)

print("adaptive threshold 을 통해 이진화 결과\n진행하기 위해 이미지창에서 스페이스바를 눌러주세요.\n")
cv_space_check()


# 4-A 이진 모폴로지 연산을 이용하여 잡음제거 및 윤곽선 스무싱 (4-B 적절한 크기와 형태의 SE, 연산의 종류 선택)
# (임의로 SE = 5*5 직사각형, 연산 종류 = 열기->닫기 진행했습니다)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
thres = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)
thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
cv2.imshow('res3', thres)

print("이진 모폴로지 연산을 이용해 잡음제거 및 윤곽선 스무싱 결과\n종료하려면 스페이스바를 눌러주세요")
cv_space_check()
exit()
