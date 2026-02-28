import cv2
import os

# Initialize webcam and face detector
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Removed CAP_DSHOW if on Mac
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get the student's ID number
face_id = input('\n 🙋‍♂️ Enter your Student ID (e.g., 101) and press Enter: ')
print("\n [INFO] Initializing face capture. Look at the camera and wait...")

count = 0
while True:
    success, img = camera.read()
    img = cv2.flip(img, 1) # Mirror image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        # Format: User.101.1.jpg
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
        cv2.imshow('Capturing Face Data', img)

    # Press 'ESC' to stop early, or it auto-stops at 50 photos
    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break
    elif count >= 50:
        break

print("\n [INFO] 50 Photos captured successfully! Cleaning up...")
camera.release()
cv2.destroyAllWindows()