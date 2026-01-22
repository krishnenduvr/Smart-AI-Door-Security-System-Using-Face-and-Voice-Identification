# import tensorflow as tf
# import numpy as np
# import cv2
# import pickle
# from tensorflow.keras import layers, models # type: ignore
# from tensorflow.keras.preprocessing import image_dataset_from_directory # type: ignore
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
# from mtcnn import MTCNN


# data=r"D:\Security System\cropped_faces"
# a=tf.keras.preprocessing.image_dataset_from_directory(data,image_size=(128,128),batch_size=32)
# b=a.class_names
# print("classes:",b)

# train_ds=tf.keras.utils.image_dataset_from_directory(
#     data, validation_split=0.1, subset="training",
#     seed=42, image_size=(128,128), batch_size=32,
#     class_names=b)

# val_ds=tf.keras.utils.image_dataset_from_directory(
#     data, validation_split=0.1, subset="validation",
#     seed=42, image_size=(128,128), batch_size=32,
#     class_names=b)


# for images, labels in train_ds.take(1):
#     print("train labels:", labels.numpy()[:10])
# for images, labels in val_ds.take(1):
#     print("val labels:", labels.numpy()[:10])

# AUTOTUNE=tf.data.AUTOTUNE
# train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds=val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# model=models.Sequential([
#     layers.Rescaling(1./255, input_shape=(128,128,3)),

#     layers.Conv2D(32,(3,3),activation='relu',padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),

#     layers.Conv2D(64,(3,3),activation='relu',padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),

#     layers.Conv2D(128,(3,3),activation='relu',padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),

#     layers.Conv2D(256,(3,3),activation='relu',padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),

#     layers.GlobalAveragePooling2D(),
#     layers.Dense(256,activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(len(b),activation='softmax')
# ])

# optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
# model.compile(optimizer=optimizer,
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])



# early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
# checkpoint = ModelCheckpoint("best_cnn_model1.h5", monitor='val_loss', save_best_only=True)
# lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)



# history=model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=50,
#     batch_size=32,
#     callbacks=[
#     # early_stop,
#     checkpoint,
#     lr_plateau]
#     )
# model.save("final_cnn_model1.h5")

# with open("class_names.pkl","wb") as f:
    # pickle.dump(b,f)
   


import tensorflow as tf
import numpy as np
import cv2
import pickle
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from mtcnn import MTCNN

# data = r"D:\Security System\cropped_faces"
data = r"D:\Security System\cropped_captured"


# Load dataset
a = tf.keras.preprocessing.image_dataset_from_directory(
    data, image_size=(128,128), batch_size=32
)
b = a.class_names
print("classes:", b)

with open("real_names.pkl", "wb") as f:
    pickle.dump(b, f)

train_ds = tf.keras.utils.image_dataset_from_directory(
    data, validation_split=0.1, subset="training",
    seed=42, image_size=(128,128), batch_size=32,
    class_names=b
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data, validation_split=0.1, subset="validation",
    seed=42, image_size=(128,128), batch_size=32,
    class_names=b
)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -------------------------------
# Fine-tuning setup
# -------------------------------

# 1. Load pre-trained backbone
model = tf.keras.models.load_model("final_cnn_model1.h5")
for layer in model.layers[:-1]:  # Freeze all layers except the last one
    layer.trainable = True
model.pop()  # remove last layer
model.add(layers.Dense(3, activation="softmax", name="dense_3class"))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
mc = ModelCheckpoint("fine_tuned_model.h5", save_best_only=True, monitor="val_accuracy")
lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[es, mc, lr])

# 6. Train classifier head
history = model.fit(
    train_ds,
    validation_data=val_ds,
    # epochs=10,
    epochs=20,
    batch_size=32,
    callbacks=[checkpoint, lr_plateau])

# -------------------------------
# Fine-tune deeper layers
# -------------------------------

# 7. Unfreeze some layers of base model
base_model.trainable = True
# for layer in base_model.layers[:100]:  # keep first 100 frozen
for layer in base_model.layers[:30]:
    layer.trainable = False

# 8. Re-compile with smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 9. Continue training (fine-tuning)
history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    # epochs=20,
    epochs=30,
    batch_size=32,
    callbacks=[mc, lr_plateau]
)

# 10. Save final fine-tuned model
model.save("final_finetuned_model.keras")

with open("real_names.pkl", "rb") as f:
    class_names = pickle.load(f)

model = tf.keras.models.load_model("fine_tuned_model.h5")
detector = MTCNN()

def predict_face(img_path):
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    x, y, w, h = faces[0]['box']   # take first face
    face = img_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, (128,128))
    face = np.expand_dims(face, axis=0)
    pred = model.predict(face) 
    idx = np.argmax(pred, axis=1)[0]
    print("Raw prediction vector:", pred[0])
    print("Argmax index:", np.argmax(pred[0]))
    print(f"Predicted: {class_names[idx]} ({100*np.max(pred[0]):.2f}% confidence)")

cap = cv2.VideoCapture(0)
print("Press SPACE to capture, ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        predict_face(frame)   # pass frame directly
        break

cap.release()
cv2.destroyAllWindows()