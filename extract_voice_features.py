import os 
import librosa
import numpy as np
import pickle

dataset_path=r"D:\Security System\augmented_voices"
save_path=r"D:\Security System\voice_features.pkl"
features=[]
labels=[]

for person in os.listdir(dataset_path):
    person_path=os.path.join(dataset_path,person)
    if not os.path.isdir(person_path):
        continue
    for file in os.listdir(person_path):
        if not file.endswith(".wav"):
            continue
        file_path=os.path.join(person_path,file)
        audio,sr=librosa.load(file_path,sr=22050)
        mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=40)
        mfcc=np.mean(mfcc.T,axis=0)
        features.append(mfcc)
        labels.append(person)
features=np.array(features)
labels=np.array(labels)
with open(save_path,'wb') as f:
    pickle.dump((features,labels),f)
print("Voice features extracted and saved")