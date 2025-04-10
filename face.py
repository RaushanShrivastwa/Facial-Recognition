import face_recognition
import cv2

# Load the reference image (the known image of the person)
known_image = face_recognition.load_image_file("use_your_imgae_or_path_of_image.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize the camera (webcam)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame from the camera
    ret, frame = video_capture.read()

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through all detected faces
    for face_encoding in face_encodings:
        # Compare the current face encoding with the known face encoding
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)

        if True in matches:
            # If the face matches, display a message
            cv2.putText(frame, "Match Found!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # If no match, display a different message
            cv2.putText(frame, "No Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
