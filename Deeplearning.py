# IMPORT LIBRARIES
import os
import numpy as np
from scipy.ndimage import imread
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam

# GET THE IMAGES
img_folder = 'images'
pos = os.path.join(img_folder, 'pos')
neg = os.path.join(img_folder, 'neg')
img_pos = os.listdir(pos)
img_neg = os.listdir(neg)
num_neg = len(img_neg)

# MERGE THE IMAGES INTO 2 CATEGORIES
images = [img_neg, img_pos]

# TRAINING DATA (EMPTY)
x = []
label = []

# GETTING DATA BOTH LABEL AND DATA
for i, val in enumerate(images):
    for img in val:
        name = os.path.join(neg, img)
        label_ = [0]
        if i == 1:
            label_ = [1]
            name = os.path.join(pos, img)
        data = imread(name).tolist()
        x.append(data)
        label.append(label_)

# PREPROCESSING THE DATA
# BY RESHAPING AND DIVIDE THE IMAGE TO GET VALUE
# BETWEEN 0 AND 1
x = np.array(x).reshape(-1, 150, 150, 1) / 255
label = np.array(label)

# CREATE MODEL
model = Sequential()
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(150, 150, 1)))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Conv2D(84, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# OPTIMIZER USING ADAM
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# COMPILING WITH LOSS AND OPTIMIZER
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

# TRAINING THE MODEL
history = model.fit(x, label, epochs=10, steps_per_epoch=5)

# SAVE MODEL
model_folder = 'model'
model_folder_exist = os.path.exists(model_folder)
if not model_folder_exist:
    os.makedirs(model_folder)

m = model.to_json()
m_name = os.path.join(model_folder, 'model.json')
weights = os.path.join(model_folder, 'weight.h5')
with open(m_name, 'w') as file:
    file.write(m)
model.save_weights(weights)
print('Saved Model')
