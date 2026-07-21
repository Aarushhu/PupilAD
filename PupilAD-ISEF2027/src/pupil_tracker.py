import cv2
import numpy as np
import pandas as pd
import time

# Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_pupil(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100,100))
    
    pupils = []
    for (fx, fy, fw, fh) in faces:
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        roi_color = frame[fy:fy+fh, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            
            # Threshold to isolate dark pupil
            _, thresh = cv2.threshold(
                eye_roi, 30, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > 20:
                    diameter = 2 * np.sqrt(area / np.pi)
                    pupils.append(diameter)
                    
                    # Draw on frame
                    (cx, cy), radius = cv2.minEnclosingCircle(largest)
                    center = (int(cx)+fx+ex, int(cy)+fy+ey)
                    cv2.circle(frame, center, int(radius), (0,255,0), 2)
    
    return np.mean(pupils) if pupils else None

def run_task(task_name, duration=120):
    """Run one cognitive task and record pupil data"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    recordings = []
    start_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"TASK: {task_name}")
    print(f"Duration: {duration} seconds")
    print(f"Press Q to skip to next task")
    print(f"{'='*50}\n")
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > duration:
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        diameter = detect_pupil(frame)
        
        if diameter:
            recordings.append({
                'timestamp': elapsed,
                'avg_diameter': diameter,
                'task': task_name
            })
        
        # Overlay UI
        remaining = int(duration - elapsed)
        cv2.rectangle(frame, (0,0), (400, 120), (0,0,0), -1)
        cv2.putText(frame, f"Task: {task_name}", (10,30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Time left: {remaining}s", (10,60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if diameter:
            cv2.putText(frame, f"Pupil: {diameter:.1f}px", (10,90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            cv2.putText(frame, "No pupil detected", (10,90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.imshow(f'PupilAD — {task_name}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return pd.DataFrame(recordings)

def run_full_session():
    """Run all 4 cognitive tasks in sequence"""
    tasks = [
        ('N-back', 120),
        ('Digit Span', 120),
        ('Stroop', 120),
        ('Semantic Fluency', 120)
    ]
    
    all_data = {}
    
    print("\n" + "="*50)
    print("    PupilAD — FULL SESSION STARTING")
    print("="*50)
    input("Press ENTER when ready...")
    
    for task_name, duration in tasks:
        input(f"\nPress ENTER to start: {task_name}")
        data = run_task(task_name, duration)
        all_data[task_name] = data
        print(f"Recorded {len(data)} frames for {task_name}")
        
        # Save after each task
        data.to_csv(f'pupil_{task_name.replace(" ","_")}.csv', index=False)
    
    print("\nSession complete! All data saved.")
    return all_data

if __name__ == "__main__":
    run_full_session()