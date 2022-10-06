import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

from numpy import ndarray

#called the CV2 package to capture the images
video_capture = cv2.VideoCapture(0)


#loaded the image from the location that are present in the system, we can also automate the process by jsut using the for loop
#also we have encoded the image using the name_encoding function
raghav_image = face_recognition.load_image_file("D:\IDE\project1\pictuers for project1\Raghav.jfif")
raghav_encoding = face_recognition.face_encodings(raghav_image)[0]

# ajay_image = face_recognition.load_image_file("D:\IDE\project1\pictuers\project1\Ajay.jfif")
# ajay_encoding = face_recognition.face_encodings(ajay_image)[0]

prabhjot_image = face_recognition.load_image_file("D:\IDE\project1\pictuers for project1\Prabhjot.jfif")
prabhjot_encoding = face_recognition.face_encodings(prabhjot_image)[0]

#here we have listed the name of the encoded faces into an array
known_face_encoding = [
    raghav_encoding,
    # ajay_encoding,
    prabhjot_encoding,
]
known_face_names = [
    "raghav",
    # "ajay",
    "prabhjot"
]

#Here we gave the location of the image clicked and the encoding version along with the image name
#This is the students list that we are going to use infuture, for now it is the duplicate copy of the known faces names
students = known_face_names.copy()

face_location = []
face_encoding = []
face_names = []
s=True

#we used this to get the correct or the real-time date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Now we will create the csv file with the writter class within
f = open(current_date+'.csv', 'w+', newline='')
Inwriter =  csv.writer(f)

#  now we will read the data from the webcam adn then resize it or basically decrease the size by the factor of 0.25 and then converting into the rgb format as opencv reads the data in the form of the black and white and face_recoginition for exactness takes the rbg component
while True:
    _,frame = video_capture.read()
    small_frame =cv2.resize(frame,(0,0), fx=0.25,fy=0.25)

    #Here we are going to find the best match from the captured face with the database predefined faces
    if s:
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings :
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

    # Now we will here enter the data to the CSV file with the students name and the time of the entry in the record
            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    Inwriter.writerow([name,current_time])

    #now here we will write the code to show the output to the user

    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit condition that the exit will happen only if we press 'q'
        break

video_capture.release()
cv2.destroyAllWindows()
f.close() #to close csv file

