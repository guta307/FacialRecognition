from flask import Flask, Response
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)