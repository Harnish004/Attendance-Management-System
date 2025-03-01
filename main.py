import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

# Folder to store registered faces
FACES_DIR = "faces"
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

# CSV file to store attendance records
ATTENDANCE_FILE = "attendance.csv"

# Load registered faces
def load_registered_faces():
    known_encodings = []
    known_names = []
    
    for filename in os.listdir(FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = face_recognition.load_image_file(os.path.join(FACES_DIR, filename))
            encoding = face_recognition.face_encodings(img)
            
            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(os.path.splitext(filename)[0])  # Use filename as name
    
    return known_encodings, known_names

# Function to register a new face
def register_face(name):
    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Registration", "Press 'S' to capture and register your face.")
    
    while True:
        ret, frame = cap.read()
        cv2.imshow("Register Face", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 'S' to save the face
            face_path = os.path.join(FACES_DIR, f"{name}.jpg")
            cv2.imwrite(face_path, frame)
            messagebox.showinfo("Success", f"Face registered successfully as {name}")
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to mark attendance
def mark_attendance(name):
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")

    # Load or create attendance file
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Time"])
        df.to_csv(ATTENDANCE_FILE, index=False)

    df = pd.read_csv(ATTENDANCE_FILE)

    # Append attendance record if not already marked today
    if not ((df["Name"] == name) & (df["Time"].str.startswith(now.strftime("%Y-%m-%d")))).any():
        new_record = pd.DataFrame([[name, date_time]], columns=["Name", "Time"])
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        messagebox.showinfo("Attendance", f"Attendance marked for {name} at {date_time}")
    else:
        messagebox.showinfo("Already Marked", f"{name}'s attendance is already marked today.")

# Function to recognize faces and mark attendance
def recognize_face():
    known_encodings, known_names = load_registered_faces()
    
    if not known_encodings:
        messagebox.showerror("Error", "No registered faces found. Please register first.")
        return

    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Face Recognition", "Press 'Q' to exit recognition mode.")

    while True:
        ret, frame = cap.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]
                mark_attendance(name)

            # Scale back up face locations since the frame was resized
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'Q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI for the system
def open_gui():
    root = tk.Tk()
    root.title("Attendance Management System")

    tk.Label(root, text="Face Recognition Attendance System", font=("Arial", 16)).pack(pady=10)

    tk.Button(root, text="Register Face", font=("Arial", 14), command=lambda: register_face(name_entry.get())).pack(pady=5)
    tk.Button(root, text="Mark Attendance", font=("Arial", 14), command=recognize_face).pack(pady=5)

    tk.Label(root, text="Enter Name:", font=("Arial", 12)).pack(pady=5)
    name_entry = tk.Entry(root, font=("Arial", 12))
    name_entry.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    open_gui()
