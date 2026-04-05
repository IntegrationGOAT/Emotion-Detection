import cv2
import os
from ultralytics import YOLO


if os.path.exists("runs/classify/train/weights/best.pt"):
    model = YOLO("runs/classify/train/weights/best.pt")
    print("Using trained emotion detection model")
else:
    print("Trained model not found. Please run train.py first to train the model on emotional data.")
    print("Running: python train.py")
    model = YOLO("yolov8n-cls.pt")
    model.train(data=".", epochs=20, imgsz=64, batch=32)
    model = YOLO("runs/classify/train/weights/best.pt")

# Emotion classes
emotions = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprised"
}

def detect_emotion_webcam():
    """Open webcam and detect emotions in real-time"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened. Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (64, 64))
        
        # Predict emotion
        results = model(frame_resized)
        
        # Get the predicted emotion
        if results:
            top1_class = results[0].probs.top1
            top1_conf = results[0].probs.top1conf
            predicted_emotion = emotions.get(top1_class, "unknown")
            confidence = float(top1_conf)
            
            # Display prediction on frame
            text = f"Emotion: {predicted_emotion} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Emotion Detection", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def detect_emotion_image(image_path):
    """Detect emotion from a single image"""
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Predict emotion
    results = model(img)
    
    if results:
        top1_class = results[0].probs.top1
        top1_conf = results[0].probs.top1conf
        predicted_emotion = emotions.get(top1_class, "unknown")
        confidence = float(top1_conf)
        
        print(f"Emotion: {predicted_emotion}, Confidence: {confidence:.2f}")
        
        # Display prediction on image
        text = f"Emotion: {predicted_emotion} ({confidence:.2f})"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        cv2.imshow("Emotion Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Uncomment one of the options below:
    
    # Option 1: Use webcam for real-time detection
    detect_emotion_webcam()
    
    # Option 2: Detect emotion from an image file
    # detect_emotion_image("path/to/image.jpg")
