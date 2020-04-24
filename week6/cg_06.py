import cv2
import numpy as np
# from msvcrt import getch

model_image = cv2.imread('input.jpg')
src = []

# 마우스로 이미지에 점을 4개 찍고 점의 좌표를 src에 저장
def mouse_handler(event, x, y, flags, param) :
    if event == cv2.EVENT_LBUTTONUP :
        img = model_image.copy()
        src.append([x, y])
        for xx, yy in src:
            cv2.circle(img, center=(xx,yy), radius=5, color=(0,0,255), thickness=-1)

        cv2.imshow('img', img)

        if len(src) == 4:
            src_np = np.array(src, dtype=np.float32)
            width = max(np.linalg.norm(src_np[1] - src_np[0]), np.linalg.norm(src_np[3] - src_np[2]))
            height = max(np.linalg.norm(src_np[3] - src_np[0]), np.linalg.norm(src_np[2] - src_np[1]))

            dst_np = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

            mat = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)
            result = cv2.warpPerspective(model_image, M=mat, dsize=(width, height))

            cv2.imshow('image', result)


cv2.namedWindow('img')
cv2.setMouseCallback("img", mouse_handler)

cv2.imshow('img', model_image)
cv2.waitKey(0)