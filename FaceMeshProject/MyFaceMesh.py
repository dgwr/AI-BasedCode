import cv2
import time
import FaceMeshModule as fmm

# cap = cv2.VideoCapture('videos/4.mp4')
cap = cv2.VideoCapture(0)
pTime = 0
detector = fmm.FaceMeshDetector()

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img)
    if len(faces) != 0:
        print(faces[0])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)