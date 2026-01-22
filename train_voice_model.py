import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_path=r"D:\Security System\voice_features.pkl"
model_path=r"D:\Security System\voice_model.pkl"
label_path=r"D:\Security System\voice_labels.pkl"
with open (data_path,"rb") as f:
    X,y=pickle.load(f)
encoder=LabelEncoder()
y_encoded=encoder.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y_encoded,test_size=0.1,random_state=42)
clf=SVC(kernel="linear",probability=True)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
with open(model_path,"wb") as f:
    pickle.dump(clf,f)
with open(label_path,"wb") as f:
    pickle.dump(encoder,f)