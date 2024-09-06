import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

# Load the pre-trained MobileNetV2 model
gesture_model = MobileNetV2(weights='imagenet')

# Parameters for stopping detection
NO_GESTURE_THRESHOLD = 50  # Number of consecutive frames with no gesture to trigger stop

def process_continuous_gesture_input(output_text):
    cap = cv2.VideoCapture(0)  # Open the webcam
    no_gesture_count = 0

    output_text.insert('end', "Starting continuous gesture recognition...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            output_text.insert('end', "Failed to capture image from webcam.\n")
            break

        # Preprocess the frame for MobileNetV2
        img = cv2.resize(frame, (224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make a prediction
        predictions = gesture_model.predict(img_array)
        gesture_result = decode_predictions(predictions, top=1)
        recognized_gesture = gesture_result[0][0][1]

        # If a gesture is recognized, reset the no_gesture_count
        if recognized_gesture not in ['nonsense', 'unidentified']:
            no_gesture_count = 0
            output_text.delete('1.0', tk.END)  # Clear previous gesture
            output_text.insert('end', f"Gesture Recognized: {recognized_gesture}\n")
        else:
            no_gesture_count += 1

        # Display the frame in a window (optional)
        cv2.putText(frame, f"Gesture: {recognized_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Gesture Recognition", frame)

        # Break the loop if the user presses 'q' or if the no_gesture_count exceeds the threshold
        if cv2.waitKey(1) & 0xFF == ord('q') or no_gesture_count > NO_GESTURE_THRESHOLD:
            break

    cap.release()
    cv2.destroyAllWindows()
    output_text.insert('end', "Gesture recognition stopped due to inactivity.\n")
