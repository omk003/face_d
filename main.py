from PIL import Image
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

# for face detection
face_cascade = cv2.CascadeClassifier(r'C:\Users\Lenovo\Downloads\archive\haarcascade_frontalface_default.xml')

# resolution of the webcam
screen_width = 1280  # try 640 if code fails
screen_height = 720

# size of the image to predict
image_width = 224
image_height = 224

# load the trained model
model = load_model(r"C:\Users\Lenovo\Downloads\transfer_learning_trained_face_cnn_model4.h5")

# the labels for the trained model
with open(r"C:\Users\Lenovo\Downloads\face-labels (1).pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {key: value for key, value in og_labels.items()}
    print(labels)

# default webcam
stream = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    grabbed, frame = stream.read()
    if not grabbed:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # try to detect faces in the webcam
    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=5)

    # for each face found
    for (x, y, w, h) in faces:
        roi_rgb = rgb[y:y + h, x:x + w]
        color = (255, 0, 0)  # in BGR
        stroke = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

        # resize the image
        resized_image = cv2.resize(roi_rgb, (image_width, image_height))
        image_array = np.array(resized_image, "uint8")
        img = image_array.reshape(1, image_width, image_height, 3)
        img = img.astype('float32')
        img /= 255

        # predict the image
        predicted_prob = model.predict(img)

        # Display the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[predicted_prob[0].argmax()]
        color = (255, 0, 255)
        stroke = 2
        cv2.putText(frame, f'({name})', (x, y - 8), font, 1, color, stroke, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Press q to break out of the loop
        break

# Cleanup
stream.release()
cv2.destroyAllWindows()
