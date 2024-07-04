from flask import Flask, request, jsonify
import face_recognition
import os

app = Flask(__name__)

def load_known_faces(directory):
    known_faces, known_names = [], []

    for filename in filter(lambda f: f.endswith(('.jpg', '.png')), os.listdir(directory)):
        path = os.path.join(directory, filename)
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(os.path.splitext(filename)[0])

    return known_faces, known_names

def recognize_face(unknown_image, known_faces, known_names):
    unknown_encodings = face_recognition.face_encodings(unknown_image)

    if not unknown_encodings:
        return "No face found in the provided image"

    face_distances = face_recognition.face_distance(known_faces, unknown_encodings[0])
    min_distance_index = face_distances.argmin()

    return known_names[min_distance_index] if face_distances[min_distance_index] < 0.6 else "Unknown"

known_faces, known_names = load_known_faces("known")

@app.route('/recognize', methods=['POST'])
def recognize():
    file = request.files.get('file')

    if not file or not file.filename:
        return jsonify({"error": "No file provided"}), 400

    try:
        unknown_image = face_recognition.load_image_file(file)
        result = recognize_face(unknown_image, known_faces, known_names)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
