import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from threading import Thread
import shutil

# Path to the folder containing images
path = 'Images_Attendance'
if not os.path.exists(path):
    os.makedirs(path)

images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)  # This returns a list of encodings
        if len(encodes) > 0:  # Check if the list is not empty
            encodeList.append(encodes[0])  # Append the first encoding, if available
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            time_now = datetime.now()
            tString = time_now.strftime('%H:%M:%S')
            dString = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tString},{dString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initialize the GUI
window = tk.Tk()
window.title("Attendance Tracking System")
window.geometry("800x600")
window.minsize(600, 400)

# Responsive window configuration
for i in range(3):
    window.grid_columnconfigure(i, weight=1)
    window.grid_rowconfigure(i, weight=1)

# Status Label
status_label = tk.Label(window, text="System Ready", fg="green")
status_label.grid(row=0, column=0, columnspan=3, sticky="ew", pady=20)

# New Window for Displaying Detection Names and Times
detected_window = tk.Toplevel(window)
detected_window.title("Detected Names and Times")
detected_window.geometry("400x400")
listbox = tk.Listbox(detected_window, width=50, height=20)
listbox.pack(pady=20)

def delete_entry():
    try:
        listbox.delete(listbox.curselection())
    except:
        messagebox.showerror("Error", "Please select a line to delete.")

delete_btn = tk.Button(detected_window, text="Delete", command=delete_entry)
delete_btn.pack()

def add_to_detected_list(name):
    time_now = datetime.now().strftime('%H:%M:%S')
    listbox.insert(tk.END, f"{name} at {time_now}")

def webcam_process():
    window.withdraw()  # Hide the window when the webcam starts
    cap = cv2.VideoCapture(0)
    detected_names_set = set()  # Set to store detected names temporarily

    try:
        while True:
            success, img = cap.read()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            current_frame_names = []

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                matchIndex = np.argmin(faceDis) if matches and np.any(matches) else -1
                name = classNames[matchIndex].upper() if matchIndex != -1 else "UNKNOWN"
                if faceDis[matchIndex] > 0.5:  # Adjust threshold as necessary
                    name = "PLEASE REMOVE MASK"

                color = (0, 255, 0) if name not in ["UNKNOWN", "PLEASE REMOVE MASK"] else (0, 0, 255)

                y1, x2, y2, x1 = [i * 4 for i in faceLoc]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # Ensure each name is only added once per frame
                if name not in ["UNKNOWN", "PLEASE REMOVE MASK"] and name not in detected_names_set:
                    detected_names_set.add(name)
                    current_frame_names.append(name)
                    add_to_detected_list(name)  # Add detected names to the list in the new window

            # Update attendance for all unique names detected in this frame
            for name in current_frame_names:
                markAttendance(name)

            cv2.imshow('webcam', img)
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('s')]:
                break

            # Clear the names set for the next frame
            detected_names_set.clear()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        window.deiconify()  # Show the window again when the webcam stops

def register_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        new_img = cv2.imread(file_path)
        if new_img is not None:
            img_name = os.path.basename(file_path)
            target_path = os.path.join(path, img_name)
            if img_name not in classNames:
                images.append(new_img)
                classNames.append(os.path.splitext(img_name)[0])
                encodeListKnown.append(findEncodings([new_img])[0])
                shutil.move(file_path, target_path)
                status_label.config(text=f"Image {img_name} registered and moved successfully", fg="blue")
            else:
                status_label.config(text="Image already registered", fg="red")
        else:
            status_label.config(text="Failed to load image", fg="red")

# Buttons
start_btn = tk.Button(window, text="Start Webcam", command=lambda: Thread(target=webcam_process).start(), bg="red", fg="white")
start_btn.grid(row=1, column=0, columnspan=3, sticky="ew", pady=10)

register_btn = tk.Button(window, text="Register Image", command=register_image, bg="green", fg="white")
register_btn.grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)

window.mainloop()
