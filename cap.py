import cv2
import requests
import json

api_url = "http://172.31.141.86:5000/recognize"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_image = frame[y:y+h, x:x+w]

        cv2.imwrite("captured_face.jpg", face_image)

        files = {'image': open('captured_face.jpg', 'rb')}
        response = requests.post(api_url, files=files)

        try:
            result = response.json()
            name = result.get('name', 'Unknown')
            print(f"Recognized person: {name}")
        except json.JSONDecodeError:
            print("Error decoding API response")

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
