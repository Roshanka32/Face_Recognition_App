import cv2
import numpy as np
import face_recognition

# Load and process Modi image
imgModi = face_recognition.load_image_file('Images_Attendance/Elon.jpg')
imgModi = cv2.cvtColor(imgModi, cv2.COLOR_BGR2RGB)
facelocModi = face_recognition.face_locations(imgModi)[0]
encodeModi = face_recognition.face_encodings(imgModi)[0]
cv2.rectangle(imgModi, (facelocModi[3], facelocModi[0]), (facelocModi[1], facelocModi[2]), (155, 0, 255), 2)

# Process and handle multiple test images
for test_image_name in ['steve.jpg', 'roshanka.jpeg']:
    imgTest = face_recognition.load_image_file(f'Images_Attendance/{test_image_name}')
    imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
    facelocTest = face_recognition.face_locations(imgTest)
    encodingsTest = face_recognition.face_encodings(imgTest)

    if len(encodingsTest) > 0:
        cv2.rectangle(imgTest, (facelocTest[0][3], facelocTest[0][0]), (facelocTest[0][1], facelocTest[0][2]),
                      (155, 0, 255), 2)

        if len(encodingsTest) > 1:
            encodeTest = encodingsTest[1]
            results = face_recognition.compare_faces([encodeModi], encodeTest)
            faceDis = face_recognition.face_distance([encodeModi], encodeTest)
            cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        (0, 0, 255), 2)
        else:
            print(f"Only one face found in {test_image_name}, using first encoding for comparison.")
            encodeTest = encodingsTest[0]
            results = face_recognition.compare_faces([encodeModi], encodeTest)
            faceDis = face_recognition.face_distance([encodeModi], encodeTest)
            cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        (0, 0, 255), 2)

        cv2.imshow(test_image_name.split('.')[0], imgTest)  # Show each test image in a separate window

cv2.imshow('Elon', imgModi)
cv2.waitKey(0)
cv2.destroyAllWindows()
