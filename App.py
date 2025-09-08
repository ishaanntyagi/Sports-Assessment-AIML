from flask import Flask, Response
import cv2

# Import your custom posture processing module
from posture_module import process_and_draw_posture

app = Flask(__name__)
video_capture = cv2.VideoCapture(0) # Initialize the camera

def generate_frames():
    """
    This function reads frames from the camera, processes them for posture,
    and yields them as a byte stream.
    """
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            # Run posture analysis and draw on the frame
            processed_frame = process_and_draw_posture(frame)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame in the multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    # This returns a special response object that streams the content
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # A simple message for the root URL
    return "Posture Detection Stream is running! Go to /video_feed to see the stream, or open index.html."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)