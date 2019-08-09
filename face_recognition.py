import cv2
import os
from scipy import misc
from keras.models import model_from_json

camera = cv2.VideoCapture(0)
haarcascade_dir = os.environ['OPEN_CV']
cascade_file = 'haarcascade_frontalface_alt.xml'
cascade_ = os.path.join(haarcascade_dir, cascade_file)
cascade = cv2.CascadeClassifier(cascade_)

yellow = (255, 255, 0)
white = tuple([255 for i in range(3)])
orange = (255, 179, 25)

text = 'Master'
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.7
size, baseline = cv2.getTextSize(text, font, font_size, 1)
width, height = size

folder = 'model'
model = 'model.json'
weight = 'weight.h5'
model_file = os.path.join(folder, model)
weight_file = os.path.join(folder, weight)

with open(model_file, 'r') as json_file:
    file = json_file.read()
    m = model_from_json(file)
    json_file.close()
m.load_weights(weight_file)

while True:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    face = cascade.detectMultiScale(frame, 1.3, 1)

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), orange, 1)
        gray_face = gray[y:y+h, x:x+w]

        try:
            predict_face = misc.imresize(gray_face, (150, 150)).reshape(1, 150, 150, 1) / 255
            pred = m.predict(predict_face)
            if pred >= 0.9:
                cv2.rectangle(frame, (x-3, y-3), (x+width+3, y-height-3), orange, -1)
                cv2.putText(frame, text, (x, y), font, font_size, white, 1)
            else:
                cv2.rectangle(frame, (x-3, y-3), (x+width+3, y-height-3), orange, -1)
                cv2.putText(frame, 'None', (x, y), font, font_size, white, 1)
        except:
            pass

    cv2.imshow('David Recognition', frame)

    interrupt = cv2.waitKey(1) & 0xFF

    if interrupt == 27:
        break

camera.release()
cv2.destroyAllWindows()
