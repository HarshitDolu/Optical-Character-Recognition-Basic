import cv2
import pytesseract

#oem is an engine mode
#page segmentation mode
pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

img=cv2.imread('images/logo.png')


img=cv2.resize(img,(512,512))

#cong = r'--oem 3 --psm 6 outputbase digits'

#boxes=pytesseract.image_to_data(img,config=cong) # for digit
#get raw information from image
#print(pytesseract.image_to_string(img))

#coordinates for box, detecting characters

himg,wimg,_=img.shape
#boxes=pytesseract.image_to_boxes(img)
#for b in boxes.splitlines():
    #print(b)
 #   b=b.split(' ')
    #print(b)
  #  x,y,w,h=int(b[1]),int(b[2]),int(b[3]),int(b[4])
   # cv2.rectangle(img,(x,himg-y),(w,himg-h),(0,0,255),3)
    #cv2.putText(img,b[0],(x,himg-y-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)




# detecting words

boxes=pytesseract.image_to_data(img)
for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b=b.split()
        print(b)
        if len(b)==12:
            x,y,w,h=int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(img, (x, y), (w+x, h+y), (0, 0, 255), 3)
            cv2.putText(img,b[11],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)





cv2.imshow("image",img)
cv2.waitKey(0)