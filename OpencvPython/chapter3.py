import cv2

img = cv2.imread("resources/lena.jpg")
print(img.shape)

imgResize = cv2.resize(img, (200, 100))
print(imgResize.shape)
imgCropped = img[0:100, 100:200]

cv2.imshow("Image", img)
cv2.imshow("Image Resize", imgResize)
cv2.imshow("Image Cropped", imgCropped)

cv2.waitKey(0)
