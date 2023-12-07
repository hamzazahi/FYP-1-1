import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load labels from coco2.names
with open("coco2.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    car_class_id = classes.index('car')  # Get the class id for 'car'

# Load video
cap = cv2.VideoCapture('test.mp4')

# Initialize time
prev_time = time.time()

# Define the ROI coordinates (you need to set these according to your ROI)
x_start, y_start, x_end, y_end = 100, 100, 700, 500  # Example coordinates

# Initialize the frame variable
frame = None

# Initialize the total detections variable
total_detections = 0

while cap.isOpened():
    current_time = time.time()
    
    # Only process new frame if enough time has passed
    if current_time - prev_time >= 5:
        # Initialize the count for new detections in this interval
        new_detections = 0

        # Reset the lists for each detection interval
        boxes = []
        confidences = []
        class_ids = []

        # Read video frame by frame
        ret, frame = cap.read()
        
        # Check if the video has ended and loop if necessary
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
            continue

        # Resize the frame to the desired display size
        frame = cv2.resize(frame, (600, 400))

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Processing detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Adjust confidence threshold if necessary
                    # Object detected
                    center_x = int(detection[0] * 600)
                    center_y = int(detection[1] * 400)
                    w = int(detection[2] * 600)
                    h = int(detection[3] * 400)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    # Check if the detection is within the ROI for counting cars
                    if x_start <= x <= x_end and y_start <= y <= y_end and class_id == car_class_id:
                        new_detections += 1  # Increment new detections count

        # Update the total detections
        total_detections += new_detections

        # Apply Non-Max Suppression to reduce overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0) if class_ids[i] == car_class_id else (255, 0, 0)  # Green for cars, red for others
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

        # Display new and total detections
        cv2.putText(frame, f"New Detections: {new_detections}", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Total Detections: {total_detections}", (5, 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Update the time of the last processed frame
        prev_time = current_time

    # If a frame has been processed, display it
    if frame is not None:
        cv2.imshow("Image", frame)

    # Break the loop
    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to exit
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
