from flask import Flask, request, jsonify,render_template 
import face_recognition
import os
from datetime import datetime

app = Flask(__name__)

attendance_records = {}

def load_known_faces(directory):
    known_faces = []
    known_names = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(os.path.splitext(filename)[0])

    return known_faces, known_names

def recognize_face(unknown_image, known_faces, known_namaes):
    unknown_encoding = face_recognition.face_encodings(unknown_image)

    if len(unknown_encoding) > 0:
        face_distances = face_recognition.face_distance(known_faces, unknown_encoding[0])
        min_distance_index = face_distances.argmin()

        if face_distances[min_distance_index] < 0.6:  
            return known_names[min_distance_index]
        else:
            return "Unknown"
    else:
        return "No face found in the provided image"

known_faces_dir = "known"

known_faces, known_names = load_known_faces(known_faces_dir)

@app.route('/recognize', methods=['POST'])
def recognize():
    print(request.files)

    if '' in request.files:
        file = request.files['']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file:
            try:
                unknown_image = face_recognition.load_image_file(file)
                result = recognize_face(unknown_image, known_faces, known_names)

                update_attendance(result)

                return jsonify({"result": result})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "No file provided"}), 400

def update_attendance(name):
    if name != "Unknown":
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if name not in attendance_records:
            attendance_records[name] = []

        attendance_records[name].append(timestamp)
@app.route('/view_records', methods=['GET'])
def view_records():
    return render_template('attendance_records.html', records=attendance_records)

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
