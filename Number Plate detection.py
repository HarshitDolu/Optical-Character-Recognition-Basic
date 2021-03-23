import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
import imutils



img=cv2.imread("images/plate1.jpg")

pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
gray=cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)



filter=cv2.bilateralFilter(gray,11,15,15)
edge=cv2.Canny(filter,170,200)

keypoints=cv2.findContours(edge.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

final=None
for c in contours:
    approx=cv2.approxPolyDP(c,10,True)
    if len(approx) == 4:
        final=approx
        x, y, w, h = cv2.boundingRect(final)
        break

p1=x
p2=y

cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)



mask = np.zeros(gray.shape, np.uint8)

new_image = cv2.drawContours(mask, [final], 0,255, -1)
nn=cv2.bitwise_and(img,img,mask=mask)
cv2.imshow("image",nn)
cv2.waitKey(0)

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]



boxes=pytesseract.image_to_boxes(cropped_image)
print(boxes)

text=''
for b in boxes.splitlines():
    print(b)
    b=b.split(' ')
    print(b)
    text=text+b[0]

print(text)
cv2.putText(img,text,(p1,p2-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
cv2.imshow("image",img)

cv2.waitKey(0)

images=[new_image,nn,cropped_image,img]
titles=['image mask','on bitwise_and','Cropped_image','Final image']
siz=len(titles)

for i in range(0,siz):
    plt.subplot(2, 4, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()