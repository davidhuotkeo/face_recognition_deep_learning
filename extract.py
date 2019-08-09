import cv2
import os
from scipy import misc

# CREATING OBJECT OF OPENCV
camera = cv2.VideoCapture(0)

# GETTING THE PATH AND HAARCASCADE FOR FACE
opencv = os.environ['OPEN_CV']
xml_file = 'haarcascade_frontalface_alt.xml'
cascade_file = os.path.join(opencv, xml_file)
cascade = cv2.CascadeClassifier(cascade_file)

# COLORS
yellow = (255, 255, 0)
white = tuple([255 for i in range(3)])

# CREATING FOLDER FOR THE IMAGES SAVED
folder = 'images'
n = os.path.join(folder, 'neg')
p = os.path.join(folder, 'pos')
img_ext = 'png'
folder_exist = os.path.exists(folder)
if not folder_exist:
    os.makedirs(folder)
    os.makedirs(n)
    os.makedirs(p)

# OPENING CAMERA LOOPING
while True:
    # GETTING NUMBER OF PICTURES IN BOTH
    # NEGATIVE AND POSITIVE IMAGE FOLDER
    number_n_img = len(os.listdir(n))
    text_picture_num_n = 'Pictures in folder: {}'.format(number_n_img)

    number_p_img = len(os.listdir(p))
    text_picture_num_p = 'Pictures in folder: {}'.format(number_p_img)

    # READ THE DATA FROM THE CAMERA
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    img_save = frame

    # GETTING X, Y, WIDTH AND HEIGHT OF THE
    # FACE DETECTION AND TURN INTO GRAYSCALE
    face = cascade.detectMultiScale(frame, 1.3, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # PUT NUMBERS OF IMAGES
    cv2.putText(frame, text_picture_num_p, (70, 50), cv2.FONT_ITALIC, 0.5, white, 1)
    cv2.putText(frame, text_picture_num_n, (70, 70), cv2.FONT_ITALIC, 0.5, white, 1)

    # X, Y, W, H OF THE FACE
    # AND DRAW RECTANGLE
    a = b = c = d = 0
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), yellow, 1)
        a, b, c, d = x, y, w, h

    # SHOW THE FRAMES
    cv2.imshow('Extract', frame)

    interrupt = cv2.waitKey(1) & 0xFF

    # IF PRESS 'ESC' ON KEYBOARD BREAK THE LOOP
    if interrupt == 27:
        break

    # IF PRESS 'SPACE BAR' ON KEYBOARD TO SAVE POSITIVE IMAGES
    if interrupt == 32:
        name = '{}.{}'.format(number_p_img, img_ext)
        file_name = os.path.join(p, name)
        gray_face = gray[b:b+d, a:a+c]
        gray_face = misc.imresize(gray_face, (150, 150))
        cv2.imwrite(file_name, gray_face)

    
    # IF PRESS 'TAB' ON KEYBOARD TO SAVE NEGATIVE IMAGES
    if interrupt == 8:
        name = '{}.{}'.format(number_n_img, img_ext)
        file_name = os.path.join(n, name)
        gray_face = gray[b:b+d, a:a+c]
        gray_face = misc.imresize(gray_face, (150, 150))
        cv2.imwrite(file_name, gray_face)

# CLOSING THE CAMERA OF OPENCV
camera.release()
cv2.destroyAllWindows()
