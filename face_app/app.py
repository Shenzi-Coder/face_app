import os
import face_recognition
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # حداکثر ۵ مگابایت

# ایجاد پوشه آپلود در صورت عدم وجود
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('known_faces', exist_ok=True)

# بارگذاری چهره‌های شناخته شده در شروع برنامه
known_encodings = []
known_names = []

def load_known_faces():
    global known_encodings, known_names
    known_encodings = []
    known_names = []
    if os.path.exists('known_faces'):
        for filename in os.listdir('known_faces'):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join('known_faces', filename)
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    # نام شخص را از نام فایل (قبل از _) استخراج می‌کنیم
                    name = filename.rsplit('_', 1)[0]
                    known_names.append(name)
                else:
                    print(f"هشدار: در فایل {filename} چهره‌ای یافت نشد.")

# بارگذاری اولیه
load_known_faces()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'فایلی ارسال نشده است'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'فایلی انتخاب نشده است'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        unknown_image = face_recognition.load_image_file(filepath)
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        if len(unknown_encodings) == 0:
            return jsonify({'result': 'no_face', 'message': 'هیچ چهره‌ای در تصویر پیدا نشد.'})

        unknown_encoding = unknown_encodings[0]

        # مقایسه با چهره‌های شناخته شده
        results = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.6)
        
        if True in results:
            # پیدا کردن نزدیک‌ترین تطابق
            face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
            best_match_index = face_distances.argmin()
            name = known_names[best_match_index]
            return jsonify({'result': 'known', 'name': name, 'message': f'این تصویر مربوط به {name} است.'})
        else:
            return jsonify({'result': 'unknown', 'message': 'این شخص ناشناس است.'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # پاک کردن فایل آپلود شده پس از پردازش
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)