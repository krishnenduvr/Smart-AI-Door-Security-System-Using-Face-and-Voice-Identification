import cv2
import numpy as np
import tensorflow as tf
import pickle
from mtcnn import MTCNN

# model=tf.keras.models.load_model("final_cnn_model1.h5")
model=tf.keras.models.load_model("fine_tuned_model.h5")

with open("real_names.pkl","rb") as f:
    class_names=pickle.load(f)

detector=MTCNN()
cap=cv2.VideoCapture(0)

# con=0.75
# con=0.5

while True:
    ret,frame=cap.read()
    if not ret:
        break
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces=detector.detect_faces(rgb)
    
    for face in faces:
        x,y,w,h=face["box"]
        x,y=max(0,x),max(0,y)

        face_crop=rgb[y:y+h, x:x+w]
        if face_crop.size==0:
            continue

        crop=cv2.resize(face_crop,(128,128))
        crop1=np.expand_dims(crop,axis=0)

        pred=model.predict(crop1)
        preds=pred[0]
        confidence=float(np.max(preds))
        label_index=int(np.argmax(preds))
        name=class_names[label_index]

        # #  if confidence>con:
        # if confidence<con:
        #     name="Unknown"
        # else:
        #      name = class_names[label_index]


        # if confidence<con:
        #     name="Unknown"
        

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,f"{name} ({confidence:.2f})",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,0),2)
        print("Predictions:", preds)
        print("Confidence:", confidence, "Label:", name)
    cv2.imshow("Smart Door Face Recognition",frame)

    key=cv2.waitKey(1)
    if key==0 or key==ord('q'):
        break
    # if cv2.waitKey(1) & 0xFF==27:
    #     break

cap.release()
cv2.destroyAllWindows()
      