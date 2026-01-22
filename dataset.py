import cv2
import os

S="captured_faces"
os.makedirs(S,exist_ok=True)

cap=cv2.VideoCapture(0)
count=0

while True:
    ret,frame=cap.read()
    if not ret:
        break
    cv2.imshow("Image Capture - Press 'c' to save, 'q' to quit", frame)
    key=cv2.waitKey(1) & 0xFF

    if key ==ord("c"):
        img_name=os.path.join(S,f"face_{count}.jpg")
        cv2.imwrite(img_name,frame)
        print(f"Saved {img_name}")
        count +=1
    elif key ==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()            
