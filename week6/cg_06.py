import cv2
import numpy as np
# from msvcrt import getch


# 스페이스바 체크 함수
# def space_check():
#     while True:
#         if ord(getch()) == 32:
#             break

# opencv 스페이스바 체크 함수
def cv_space_check():
    while True:
        if cv2.waitKey(1)==32: break


model_image = cv2.imread('input.jpg')
print("이미지 읽기 완료")

# 그림의 꼭짓점 좌표를 저장할 변수 (순서 : 왼쪽 위-오른쪽 위-왼쪽 아래-오른쪽 아래)
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

        # 빨간 원이 네개가 되면
        if len(src) == 4:
            print(src[3])
            # float32 np array 타입으로 저장
            src_np = np.array(src, dtype=np.float32)

            # 두 점 사이의 거리를 구하는 np.linalg.norm 함수를 사용하여 표시한 점 중 더 긴 가로/세로를 찾아낸다
            # width = int(max(np.linalg.norm(src_np[1] - src_np[0]), np.linalg.norm(src_np[3] - src_np[2])))
            # height = int(max(np.linalg.norm(src_np[3] - src_np[0]), np.linalg.norm(src_np[2] - src_np[1])))

            # template 직사각형의 좌표를 생성
            # dst_np = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

            # 변환행렬 생성
            print("변환행렬 생성을 위해 이미지창에서 스페이스바를 눌러주세요.\n")
            cv_space_check()

            mat = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)
            print("변환행렬\n",mat)

            print("\n결과영상 출력을 위해 이미지창에서 스페이스바를 눌러주세요.\n")
            cv_space_check()

            # 변환영상
            result = cv2.warpPerspective(model_image, M=mat, dsize=(400, 100))
            cv2.imshow('res', result)
            print("결과 영상 출력 완료\n종료하시려면 이미지창에서 스페이스바를 눌러주세요.")


cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse_handler)

print("\n왼쪽 위 점을 클릭해 주세요.")
cv2.imshow('img', model_image)
cv_space_check()
cv2.destroyAllWindows()
exit()
