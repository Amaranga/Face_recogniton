import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# creating a list of images
path = 'Training_images'
images = []
class_Names = []
my_List = os.listdir(path)
print(my_List)

# looping throught the list to get the class names
for cl in my_List:
    current_Img = cv2.imread(f'{path}/{cl}')
    images.append(current_Img)
    class_Names.append(os.path.splitext(cl)[0])
print(class_Names)

# creating a function to encode through all the images
def find_Encodings(images):
    encode_List = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_List.append(encode)
    return encode_List

# mark attendance
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        my_Data_List = f.readlines()

        nameList = []
        for line in my_Data_List:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            date_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date_string}')

encode_List_Known = find_Encodings(images)
print('encoding complete with all images:', len(encode_List_Known))

# cv2 uses the webcam as serves us a test image

cap = cv2.VideoCapture(0)     # initialize the webcam

# while loop to get the each frame one by one
while True:
    success, img = cap.read()
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faces_current_frame = face_recognition.face_locations(imgS)
    encodes_current_frame = face_recognition.face_encodings(imgS, faces_current_frame)

    for encodeFace, faceLoc in zip(encodes_current_frame, faces_current_frame):
        matches = face_recognition.compare_faces(encode_List_Known, encodeFace)
        face_distance = face_recognition.face_distance(encode_List_Known, encodeFace)
# print(faceDis)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = class_Names[match_index].lower()
# print(name)
            y1, x2, y2, x1 = faceLoc                                      # storiing the return values of faceloc function
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4               # here we multiplied wiht 4 to get original size
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)       
        #markAttendance(name)
    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == 13:
        break
    #cv2.waitKey(1)
#markAttendance(name)