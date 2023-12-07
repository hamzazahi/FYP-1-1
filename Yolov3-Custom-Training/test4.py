import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov4_vehicle_orientation_74500.weights", "yolov4_vehicle_orientation.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load labels from vehicle_orientation.names
with open("vehicle_orientation.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the class names you want to detect (front-facing vehicles)
classes_to_detect = ["car_front"]

# Get the indices of the classes you want to detect
class_indices_to_detect = [classes.index(class_name) for class_name in classes_to_detect]

# Open four video capture objects with different file names
cap1 = cv2.VideoCapture('test.mp4')
cap2 = cv2.VideoCapture('test2.mp4')
cap3 = cv2.VideoCapture('test3.mp4')
cap4 = cv2.VideoCapture('test4.mp4')

# Set the window size for the output display
window_width = 900
window_height = 500

# Create a window for the output display
cv2.namedWindow('Split Screen', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Split Screen', window_width, window_height)

def process_frame(cap, net, output_layers, class_indices_to_detect, scale_factor, size, mean, swapRB):
    ret, frame = cap.read()
    if not ret:
        return None, 0
    frame = cv2.resize(frame, (window_width // 2, window_height // 2))
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, scale_factor, size, mean, swapRB, crop=False)
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
            if confidence > 0.3 and class_id in class_indices_to_detect:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
    indexes = indexes.flatten() if isinstance(indexes, np.ndarray) else indexes
    for i in indexes:
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame, len(indexes)

while True:
    # Process each frame
    frame1, count1 = process_frame(cap1, net, output_layers, class_indices_to_detect, 1/255.0, (512, 512), (0, 0, 0), True)
    frame2, count2 = process_frame(cap2, net, output_layers, class_indices_to_detect, 1/255.0, (512, 512), (0, 0, 0), True)
    frame3, count3 = process_frame(cap3, net, output_layers, class_indices_to_detect, 1/255.0, (512, 512), (0, 0, 0), True)
    frame4, count4 = process_frame(cap4, net, output_layers, class_indices_to_detect, 1/255.0, (512, 512), (0, 0, 0), True)
    
    # Check if all frames are captured
    if not (frame1 is not None and frame2 is not None and frame3 is not None and frame4 is not None):
        break

    # Combine frames into a split-screen layout
    top_row = np.hstack((frame1, frame2))
    bottom_row = np.hstack((frame3, frame4))
    split_screen = np.vstack((top_row, bottom_row))

    # Display the detection count with labels on each quadrant
    cv2.putText(split_screen, f"Traffic 1: {count1} Vehicles", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(split_screen, f"Traffic 2: {count2} Vehicles", (window_width // 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(split_screen, f"Traffic 3: {count3} Vehicles", (10, window_height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(split_screen, f"Traffic 4: {count4} Vehicles", (window_width // 2 + 10, window_height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the split-screen output
    cv2.imshow('Split Screen', split_screen)

    # Break the loop if 'ESC' is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release video capture objects and close windows
cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
