from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import sqlite3
from datetime import datetime
import os
import numpy as np
import base64
import pandas as pd  # <-- Required for Smart Insights & Charts

app = Flask(__name__)

# Ensure the dataset folder exists
os.makedirs('dataset', exist_ok=True)

# --- 1. DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS Students (student_id TEXT PRIMARY KEY, name TEXT NOT NULL, date_added TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS Attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, date TEXT, time TEXT, status TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- 2. LOAD AI MODELS ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists('trainer.yml'):
    try:
        recognizer.read('trainer.yml')
        print("🧠 AI Brain loaded successfully.")
    except Exception as e:
        print("⚠️ Warning: trainer.yml is empty or corrupted. It will be recreated on next training.")

# --- 3. STUDENT MANAGEMENT (ADD & DELETE) ---
@app.route('/save_face_frame', methods=['POST'])
def save_face_frame():
    data = request.json
    student_id = data['id']
    frame_num = data['frame']
    
    img_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.imwrite(f"dataset/User.{student_id}.{frame_num}.jpg", gray[y:y+h, x:x+w])
        return jsonify({"face_found": True})
    return jsonify({"face_found": False})

@app.route('/train_ai', methods=['POST'])
def train_ai():
    try:
        from PIL import Image
        data = request.json
        student_id = data['id']
        student_name = data['name']
        
        print(f"\n🧠 Starting AI Training for ID: {student_id}...")

        conn = sqlite3.connect('attendance.db')
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("INSERT OR REPLACE INTO Students (student_id, name, date_added) VALUES (?, ?, ?)", (student_id, student_name, today))
        conn.commit()
        conn.close()

        imagePaths = [os.path.join('dataset', f) for f in os.listdir('dataset') if f.endswith('.jpg')]
        if len(imagePaths) == 0: 
            return jsonify({"status": "error", "message": "No images found."})

        faceSamples, ids = [], []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id_num = int(os.path.split(imagePath)[-1].split(".")[1])
            faceSamples.append(img_numpy)
            ids.append(id_num)
            
        recognizer.train(faceSamples, np.array(ids))
        recognizer.write('trainer.yml')
        recognizer.read('trainer.yml') 
        
        print("🏆 AI Brain Successfully Trained!")
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/delete_student', methods=['POST'])
def delete_student():
    data = request.json
    student_id = data['id']
    
    conn = sqlite3.connect('attendance.db')
    conn.execute("DELETE FROM Students WHERE student_id=?", (student_id,))
    conn.execute("DELETE FROM Attendance WHERE student_id=?", (student_id,))
    conn.commit()
    conn.close()
    
    for file in os.listdir('dataset'):
        if file.startswith(f"User.{student_id}."):
            os.remove(os.path.join('dataset', file))
            
    imagePaths = [os.path.join('dataset', f) for f in os.listdir('dataset') if f.endswith('.jpg')]
    if len(imagePaths) > 0:
        from PIL import Image
        faceSamples, ids = [], []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            faceSamples.append(np.array(PIL_img, 'uint8'))
            ids.append(int(os.path.split(imagePath)[-1].split(".")[1]))
        recognizer.train(faceSamples, np.array(ids))
        recognizer.write('trainer.yml')
        recognizer.read('trainer.yml')
    else:
        if os.path.exists('trainer.yml'): os.remove('trainer.yml')

    return jsonify({"status": "success"})

@app.route('/get_students_db')
def get_students_db():
    conn = sqlite3.connect('attendance.db')
    records = conn.execute("SELECT student_id, name, date_added FROM Students").fetchall()
    conn.close()
    return jsonify(records)

# --- 4. SECURE ATTENDANCE LOGIC & VIDEO FEED ---
def log_attendance(student_id):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    now = datetime.now()
    date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    cursor.execute("SELECT * FROM Attendance WHERE student_id=? AND date=?", (student_id, date_str))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO Attendance (student_id, date, time, status) VALUES (?, ?, ?, 'Present')", (student_id, date_str, time_str))
        conn.commit()
    conn.close()

def generate_frames():
    camera = cv2.VideoCapture(0) # Generic for both Windows/Mac
    while True:
        success, frame = camera.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            try:
                id_num, distance = recognizer.predict(gray[y:y+h, x:x+w])
                
                # If distance is low enough, it's a match
                if distance < 65:
                    conn = sqlite3.connect('attendance.db')
                    res = conn.execute("SELECT name FROM Students WHERE student_id=?", (str(id_num),)).fetchone()
                    conn.close()
                    
                    if res:
                        name = res[0]
                        # Calculate high-accuracy confidence %
                        conf_percent = round(120 - distance, 1)
                        if conf_percent > 99.8: conf_percent = 99.8
                        
                        display_text = f"{name} ({conf_percent}%)"
                        color = (0, 255, 0) # Green Box
                        
                        log_attendance(str(id_num))
                    else:
                        display_text = "Unknown"
                        color = (0, 0, 255) # Red Box
                else:
                    display_text = "Unknown"
                    color = (0, 0, 255)
            except:
                display_text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# --- NEW: Manual Snapshot Route for Audio/Voice Feedback ---
@app.route('/mark_attendance_live', methods=['POST'])
def mark_attendance_live():
    if not os.path.exists('trainer.yml'):
        return jsonify({"status": "error", "message": "AI Model not trained yet."})

    data = request.json.get('image')
    if not data: return jsonify({"status": "error", "message": "No image data provided"})
    
    try:
        header, encoded = data.split(",", 1)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected in frame"})

        for (x, y, w, h) in faces:
            id_num, distance = recognizer.predict(gray[y:y+h, x:x+w])
            
            if distance < 65:
                conn = sqlite3.connect('attendance.db')
                res = conn.execute("SELECT name FROM Students WHERE student_id=?", (str(id_num),)).fetchone()
                conn.close()
                
                if res:
                    name = res[0]
                    conf_percent = round(120 - distance, 1)
                    if conf_percent > 99.8: conf_percent = 99.8
                    confidence_str = f"{conf_percent}%"
                    
                    log_attendance(str(id_num))
                    return jsonify({
                        "status": "success", 
                        "message": f"{name} marked present", 
                        "name": name, 
                        "confidence": confidence_str
                    })

        return jsonify({"status": "error", "message": "Warning. Face not recognized."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# --- 5. WEB ROUTES ---
@app.route('/')
@app.route('/login.html')
def login(): return render_template('login.html')

@app.route('/index.html')
def dashboard(): return render_template('index.html')

@app.route('/admin.html')
def admin(): return render_template('admin.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_today_attendance')
def get_today_attendance():
    conn = sqlite3.connect('attendance.db')
    records = conn.execute("SELECT Students.student_id, Students.name, Attendance.time FROM Attendance JOIN Students ON Attendance.student_id = Students.student_id WHERE Attendance.date = date('now') ORDER BY Attendance.time DESC").fetchall()
    conn.close()
    return jsonify(records)


# ================= PANDAS ANALYTICS & SMART INSIGHTS =================

@app.route('/api/chart_today_pie')
def chart_today_pie():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT student_id) FROM Attendance WHERE date=?", (today,))
    present_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM Students")
    total_enrolled = c.fetchone()[0]
    conn.close()

    absent_count = max(0, total_enrolled - present_count)
    return jsonify({"present": present_count, "absent": absent_count})

@app.route('/api/chart_weekly_bar')
def chart_weekly_bar():
    conn = sqlite3.connect("attendance.db")
    df = pd.read_sql_query("""
        SELECT date, COUNT(DISTINCT student_id) as count 
        FROM Attendance 
        GROUP BY date 
        ORDER BY date DESC 
        LIMIT 7
    """, conn)
    conn.close()

    if df.empty: return jsonify({"dates": [], "counts": []})

    df = df.sort_values('date')
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %d')

    return jsonify({"dates": df['date'].tolist(), "counts": df['count'].tolist()})

@app.route('/api/heatmap_data')
def heatmap_data():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT date, COUNT(DISTINCT student_id) FROM Attendance GROUP BY date")
    data = c.fetchall()
    conn.close()
    result = {row[0]: row[1] for row in data}
    return jsonify(result)

@app.route('/api/smart_insights')
def smart_insights():
    try:
        conn = sqlite3.connect("attendance.db")
        # JOIN to map IDs to Names for Pandas
        df = pd.read_sql_query("""
            SELECT a.date, s.name 
            FROM Attendance a 
            JOIN Students s ON a.student_id = s.student_id
        """, conn)
        
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM Students")
        total_enrolled = c.fetchone()[0]
        conn.close()

        if df.empty or total_enrolled == 0:
            return jsonify(["⏳ Waiting for more data to generate insights..."])

        insights = []
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate individual student attendance percentages
        total_working_days = df['date'].nunique()
        student_counts = df.groupby('name')['date'].nunique().reset_index(name='days_present')
        student_counts['attendance_percent'] = (student_counts['days_present'] / total_working_days) * 100

        # --- SMART INSIGHT 1: BEST PERFORMING STUDENT ---
        if not student_counts.empty:
            best_student = student_counts.loc[student_counts['attendance_percent'].idxmax()]
            if best_student['attendance_percent'] >= 75:
                insights.append(f"🏆 Best performing student: <b style='color:#00f2fe;'>{best_student['name']}</b> ({best_student['attendance_percent']:.0f}%).")

        # --- SMART INSIGHT 2: WEEKLY TREND ---vvjvvl;
        now = pd.Timestamp.now()
        df['week'] = df['date'].dt.isocalendar().week
        current_week = now.isocalendar().week
        last_week = current_week - 1 if current_week > 1 else 52
        
        this_week_data = df[df['week'] == current_week]
        last_week_data = df[df['week'] == last_week]
        
        this_week_avg = this_week_data.groupby('date')['name'].nunique().mean() if not this_week_data.empty else 0
        last_week_avg = last_week_data.groupby('date')['name'].nunique().mean() if not last_week_data.empty else 0
        
        if last_week_avg > 0:
            change = ((this_week_avg - last_week_avg) / last_week_avg) * 100
            if change > 0:
                insights.append(f"📈 Attendance improved by <b style='color:#00e676;'>{abs(change):.1f}%</b> this week!")
            elif change < 0:
                insights.append(f"📉 Attendance dropped by <b style='color:#ff4c4c;'>{abs(change):.1f}%</b> this week.")
        elif this_week_avg > 0 and last_week_avg == 0:
             insights.append("📈 Attendance has strongly improved this week!")

        # --- SMART INSIGHT 3: LOW ATTENDANCE ALERTS ---
        low_students = student_counts[student_counts['attendance_percent'] < 50]
        for index, row in low_students.iterrows():
            insights.append(f"⚠️ Low attendance alert for <b style='color:#f2c94c;'>{row['name']}</b> ({row['attendance_percent']:.0f}%).")

        if not insights:
            insights.append("🤖 AI is actively monitoring student patterns...")

        return jsonify(insights[:4])
    
    except Exception as e:
        print(f"Insight Error: {e}")
        return jsonify(["🤖 AI Engine is calibrating data..."])
    

# ================= STUDENT PROFILE API =================

@app.route('/dataset/<path:filename>')
def serve_dataset_image(filename):
    """Allows the frontend to securely load the student's saved AI training face."""
    return send_from_directory('dataset', filename)

@app.route('/api/student_profile/<student_id>')
def student_profile(student_id):
    """Fetches the student's photo, stats, and recent history for the profile popup."""
    try:
        conn = sqlite3.connect("attendance.db")
        c = conn.cursor()

        # 1. Get Name
        c.execute("SELECT name FROM Students WHERE student_id=?", (student_id,))
        student_row = c.fetchone()
        if not student_row:
            return jsonify({"error": "Student not found"})
        name = student_row[0]

        # 2. Get Recent History (Last 5 scans)
        c.execute("SELECT date, time FROM Attendance WHERE student_id=? ORDER BY date DESC, time DESC LIMIT 5", (student_id,))
        history_data = c.fetchall()
        history = [{"date": r[0], "time": r[1]} for r in history_data]

        # 3. Calculate Stats
        c.execute("SELECT COUNT(DISTINCT date) FROM Attendance")
        total_working_days = c.fetchone()[0]
        if total_working_days == 0: total_working_days = 1

        c.execute("SELECT COUNT(DISTINCT date) FROM Attendance WHERE student_id=?", (student_id,))
        present_days = c.fetchone()[0]

        attendance_percent = round((present_days / total_working_days) * 100, 1)
        conn.close()

        # 4. Find the first training photo for their profile picture
        photo_url = "https://via.placeholder.com/150?text=No+Photo"
        if os.path.exists('dataset'):
            for file in os.listdir('dataset'):
                if file.startswith(f"User.{student_id}."):
                    photo_url = f"/dataset/{file}"
                    break # Just grab the first photo we find

        return jsonify({
            "id": student_id,
            "name": name,
            "attendance_percent": attendance_percent,
            "present_days": present_days,
            "history": history,
            "photo": photo_url
        })
    except Exception as e:
        print(f"Profile Error: {e}")
        return jsonify({"error": str(e)})
    

# ================= SECURE ADMIN FACE LOGIN =================

# Create the CLAHE object globally at the top of your app.py if you haven't already
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

@app.route('/api/face_login', methods=['POST'])
def face_login():
    """Verifies a webcam snapshot against the trained AI model for Admin access."""
    if not os.path.exists('trainer.yml'):
        return jsonify({"status": "error", "message": "System not trained."})

    data = request.json.get('image')
    if not data: return jsonify({"status": "error", "message": "No image data"})
    
    try:
        header, encoded = data.split(",", 1)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use slightly looser detection for login to catch fast movements
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected in frame"})

        for (x, y, w, h) in faces:
            # 1. Crop the face
            face_crop = gray[y:y+h, x:x+w]
            
            # 2. Apply the exact same CLAHE lighting fix used in train.py
            face_crop = clahe.apply(face_crop)
            
            # 3. Resize to the exact same 200x200 grid used in train.py
            face_crop = cv2.resize(face_crop, (200, 200))
            
            # 4. Ask the AI to predict
            id_num, distance = recognizer.predict(face_crop)
            
            # We can use a slightly higher threshold (e.g., 75) because the 
            # Javascript frontend is now looping through multiple frames.
            if distance < 75: 
                conn = sqlite3.connect('attendance.db')
                
                # Check normal students table
                res = conn.execute("SELECT name FROM Students WHERE student_id=?", (str(id_num),)).fetchone()
                
                # Also check if it's the hardcoded Admin ID (9999) from the login page enrollment
                if id_num == 9999:
                    res = ("System Admin",)
                    
                conn.close()
                
                if res:
                    name = res[0]
                    # Authorized names check
                    if "admin" in name.lower() or "govind" in name.lower():
                        return jsonify({
                            "status": "success", 
                            "message": f"Welcome back, {name}", 
                            "name": name
                        })
                    else:
                        return jsonify({"status": "error", "message": "Unauthorized. Staff only."})

        return jsonify({"status": "error", "message": "Access Denied: Unrecognized Face"})
    except Exception as e:
        print(f"Login API Error: {e}")
        return jsonify({"status": "error", "message": str(e)})
# ==========================================
# 📊 ADMIN EXPORT & ANALYTICS APIs
# ==========================================

@app.route('/api/admin_records', methods=['GET'])
def admin_records():
    """Fetches historical attendance based on a specific date filter"""
    date_filter = request.args.get('date', '')
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    
    if date_filter:
        c.execute("SELECT a.date, s.student_id, s.name, a.time FROM Attendance a JOIN Students s ON a.student_id = s.student_id WHERE a.date = ? ORDER BY a.time DESC", (date_filter,))
    else:
        # If no date selected, show the last 100 entries globally
        c.execute("SELECT a.date, s.student_id, s.name, a.time FROM Attendance a JOIN Students s ON a.student_id = s.student_id ORDER BY a.date DESC, a.time DESC LIMIT 100")
        
    records = [{"date": row[0], "id": row[1], "name": row[2], "time": row[3]} for row in c.fetchall()]
    conn.close()
    return jsonify(records)


@app.route('/api/export_csv', methods=['GET'])
def export_csv():
    """Generates an immediate Excel/CSV file download for the administration"""
    date_filter = request.args.get('date', '')
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    
    if date_filter:
        c.execute("SELECT a.date, s.student_id, s.name, a.time FROM Attendance a JOIN Students s ON a.student_id = s.student_id WHERE a.date = ? ORDER BY a.time DESC", (date_filter,))
    else:
        c.execute("SELECT a.date, s.student_id, s.name, a.time FROM Attendance a JOIN Students s ON a.student_id = s.student_id ORDER BY a.date DESC, a.time DESC")
        
    rows = c.fetchall()
    conn.close()

    def generate():
        # CSV Header
        yield 'Date,Student ID,Student Name,Time IN\n'
        # CSV Data Rows
        for row in rows:
            yield f"{row[0]},{row[1]},{row[2]},{row[3]}\n"

    # Force the browser to download it as a file
    filename = f"Attendance_Export_{date_filter}.csv" if date_filter else "Attendance_Global_Export.csv"
    return Response(generate(), mimetype='text/csv', headers={"Content-Disposition": f"attachment; filename={filename}"})


if __name__ == "__main__":
    app.run(debug=True, port=8000)