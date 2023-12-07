import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("yolov4_vehicle_orientation_74500.weights", "yolov4_vehicle_orientation.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] if isinstance(i, np.ndarray) else layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define the class names you want to detect
classes_to_detect = ["cycle_front", "motorcycle_front", "car_front", "bus_front", "truck_front"]

# Load labels from vehicle_orientation.names
with open("vehicle_orientation.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the indices of the classes you want to detect
class_indices_to_detect = [classes.index(class_name) for class_name in classes_to_detect]

# Load video
cap = cv2.VideoCapture('test3.mp4')

# Initialize time and frame
prev_time = time.time()
ret, frame = cap.read()
frame = cv2.resize(frame, (1280, 720)) if ret else None

while True:
    current_time = time.time()
    if current_time - prev_time >= 1 and frame is not None:
        # Read video frame by frame
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
            continue

        # Resize frame to 400x400
        frame = cv2.resize(frame, (1280, 720))

        height, width, channels = frame.shape

        # Detecting specific classes
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id in class_indices_to_detect:
                    # Specific class detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes without labels
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        prev_time = current_time

    if frame is not None:
        # Show the frame (either the old or the new snapshot)
        cv2.imshow("Image", frame)

    # Break the loop
    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
