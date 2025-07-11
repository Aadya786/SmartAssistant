import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import threading
import csv
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Mediapipe modules
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Thresholds
slouch_distance = 180
focus_distance = 0.025

# Global counters
focused_frames = 0
not_focused_frames = 0
good_posture_frames = 0
slouching_frames = 0
running_session = False

# Font for OpenCV window using PIL
FONT_PATH = "C:/Windows/Fonts/times.ttf"  # Update if you're on Mac/Linux
FONT_SIZE = 24
try:
    assistant_font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except:
    assistant_font = ImageFont.load_default()

def run_session():
    global focused_frames, not_focused_frames, good_posture_frames, slouching_frames, running_session
    focused_frames = 0
    not_focused_frames = 0
    good_posture_frames = 0
    slouching_frames = 0
    running_session = True

    cap = cv2.VideoCapture(0)
    baseline_left = None
    baseline_right = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
            mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:

        while cap.isOpened() and running_session:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            pose_results = pose.process(rgb_frame)
            face_results = face_mesh.process(rgb_frame)

            rgb_frame.flags.writeable = True
            image_height, image_width, _ = frame.shape

            # Posture detection
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                left_eye = landmarks[2]
                left_shoulder = landmarks[11]

                vertical_distance_pixels = (left_shoulder.y - left_eye.y) * image_height

                if vertical_distance_pixels >= slouch_distance:
                    posture = "Good Posture"
                    posture_color = (0, 255, 0)
                    good_posture_frames += 1
                else:
                    posture = "Slouching"
                    posture_color = (0, 0, 255)
                    slouching_frames += 1
            else:
                posture = ""
                posture_color = (255, 255, 255)

            # Focus detection
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                try:
                    left_iris_x = face_landmarks.landmark[474].x
                    right_iris_x = face_landmarks.landmark[469].x
                except IndexError:
                    continue

                if baseline_left is None:
                    baseline_left = left_iris_x
                if baseline_right is None:
                    baseline_right = right_iris_x

                left_offset = abs(left_iris_x - baseline_left)
                right_offset = abs(right_iris_x - baseline_right)

                if left_offset < focus_distance and right_offset < focus_distance:
                    focus_status = "Focused"
                    focus_color = (0, 255, 0)
                    focused_frames += 1
                else:
                    focus_status = "Not Focused"
                    focus_color = (0, 0, 255)
                    not_focused_frames += 1
            else:
                focus_status = ""
                focus_color = (255, 255, 255)

            # Draw using PIL
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            if posture:
                draw.text((30, 100), f"Posture: {posture}", font=assistant_font, fill=posture_color[::-1])
            if focus_status:
                draw.text((30, 140), f"Focus: {focus_status}", font=assistant_font, fill=focus_color[::-1])
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            cv2.imshow("Smart Assistant Session", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    show_summary()

def save_session_data(focus_percent, posture_percent):
    with open("session_history.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, focus_percent, posture_percent])

def show_summary():
    total_focus = focused_frames + not_focused_frames
    total_posture = good_posture_frames + slouching_frames

    focus_percent = int((focused_frames / total_focus) * 100) if total_focus else 0
    posture_percent = int((good_posture_frames / total_posture) * 100) if total_posture else 0

    save_session_data(focus_percent, posture_percent)

    summary_window = tk.Toplevel(root, bg="#f9f9f9")
    summary_window.title("Session Summary")
    summary_window.geometry("300x150")

    tk.Label(summary_window, text="📝 Session Summary", font=("Times New Roman", 16, "bold"), bg="#f9f9f9").pack(pady=10)
    tk.Label(summary_window, text=f"🎯 Focus: {focus_percent}%", font=("Times New Roman", 14), bg="#f9f9f9").pack(pady=5)
    tk.Label(summary_window, text=f"🪑 Posture: {posture_percent}%", font=("Times New Roman", 14), bg="#f9f9f9").pack(pady=5)

def view_history():
    try:
        with open("session_history.csv", mode="r") as file:
            reader = csv.reader(file)
            history = [f"{row[0]}\n🎯 Focus: {row[1]}% | 🪑 Posture: {row[2]}%" for row in reader]
    except FileNotFoundError:
        history = []

    history_window = tk.Toplevel(root, bg="#f9f9f9")
    history_window.title("Session History")
    history_window.geometry("400x300")

    tk.Label(history_window, text="📊 Recent Sessions", font=("Times New Roman", 16, "bold"), bg="#f9f9f9").pack(pady=10)

    if history:
        for entry in history[-5:]:
            tk.Label(history_window, text=entry, font=("Times New Roman", 13), justify="left", bg="#f9f9f9").pack(pady=4)
    else:
        tk.Label(history_window, text="No session history found.", font=("Times New Roman", 13), bg="#f9f9f9").pack(pady=10)

def start_session():
    session_thread = threading.Thread(target=run_session)
    session_thread.start()
    end_button.config(state="normal")

def end_session():
    global running_session
    running_session = False
    end_button.config(state="disabled")

def exit_app():
    root.destroy()

# Create main window
root = tk.Tk()
root.title("Smart Assistant")
root.geometry("300x350")
root.configure(bg="#f9f9f9")

# Set up ttk style for rounded buttons
style = ttk.Style()
style.theme_use("default")
style.configure("TButton", font=("Times New Roman", 14), padding=10, relief="flat")
style.map("TButton",
          foreground=[("active", "#000")],
          background=[("active", "#dcdcdc")])

# UI Title
label = tk.Label(root, text="🤖 Smart Assistant", font=("Times New Roman", 18, "bold"), bg="#f9f9f9")
label.pack(pady=(20, 10))

# Buttons
start_button = ttk.Button(root, text="▶ Start Session", command=start_session)
start_button.pack(pady=5)

end_button = ttk.Button(root, text="⏹ End Session", command=end_session, state="disabled")
end_button.pack(pady=5)

history_button = ttk.Button(root, text="📈 View History", command=view_history)
history_button.pack(pady=5)

exit_button = ttk.Button(root, text="❌ Exit", command=exit_app)
exit_button.pack(pady=5)

root.mainloop()
