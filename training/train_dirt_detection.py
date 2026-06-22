from google.colab import drive
drive.mount('/content/drive')

!unzip -q -o /content/drive/MyDrive/dataset_v4.zip -d /content/data_v4

import os, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
import cv2

DATA = '/content/data_v4/dataset'
RAW  = os.path.join(DATA, 'raw')
IMG  = 224
CLASSES      = ['clean', 'slightly_dirty', 'dirty']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

PANEL_PTS = np.float32([
    [550,    35],
    [2140,  140],
    [2300, 1115],
    [535,  1290],
])
WARP_W, WARP_H = 600, 400
DST_PTS = np.float32([
    [0,      0],
    [WARP_W, 0],
    [WARP_W, WARP_H],
    [0,      WARP_H],
])
M = cv2.getPerspectiveTransform(PANEL_PTS, DST_PTS)

def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(img_rgb, M, (WARP_W, WARP_H))

def load_img(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess(img)
    img = cv2.resize(img, (IMG, IMG))
    return img.astype(np.float32) / 255.0

df = pd.read_csv(os.path.join(DATA, 'manifest.csv'))
df['path'] = df.apply(lambda r: os.path.join(RAW, r['session_id'], r['class'], r['filename']), axis=1)
df = df[df['path'].apply(os.path.exists)].reset_index(drop=True)
df = df[df['class'].isin(CLASSES)].reset_index(drop=True)
df = df[~df['session_id'].str.contains('aux', case=False)].reset_index(drop=True)

df_tr, df_val = train_test_split(
    df,
    test_size=0.12,
    stratify=df['class'],
    random_state=42,
)

def load_set(d: pd.DataFrame):
    X = np.stack([load_img(p) for p in d['path']])
    y = d['class'].map(CLASS_TO_IDX).values
    return X, y

X_tr,  y_tr  = load_set(df_tr)
X_val, y_val = load_set(df_val)

aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
])

base = MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights='imagenet')
base.trainable = False

model = models.Sequential([
    layers.Input((IMG, IMG, 3)),
    aug,
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(CLASSES), activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=6, restore_best_weights=True
)

h1 = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=16,
    callbacks=[es],
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

h2 = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=16,
    callbacks=[es],
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open('/content/dirt_detection_v4_final.tflite', 'wb') as f:
    f.write(tflite_model)

!cp /content/dirt_detection_v4_final.tflite /content/drive/MyDrive/dirt_detection_v4_final.tflite
print("Saved: dirt_detection_v4_final.tflite")