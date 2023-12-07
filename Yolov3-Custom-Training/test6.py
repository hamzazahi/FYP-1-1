import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("yolov4_vehicle_orientation_74500.weights", "yolov4_vehicle_orientation.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load labels from vehicle_orientation.names
with open("vehicle_orientation.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the class names you want to detect (front-facing vehicles)
classes_to_detect = ["car_front"]

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

# Load traffic light images
red_light = cv2.imread("C:\\Users\\azaan\\OneDrive\\Desktop\\New folder\\Yolov3-Custom-Training\\traffic light\\red.png")
green_light = cv2.imread("C:\\Users\\azaan\\OneDrive\\Desktop\\New folder\\Yolov3-Custom-Training\\traffic light\\green.png")
yellow_light = cv2.imread("C:\\Users\\azaan\\OneDrive\\Desktop\\New folder\\Yolov3-Custom-Training\\traffic light\\yellow.png")

# Resize traffic light images to fit on the screen
red_light = cv2.resize(red_light, (100, 100))
green_light = cv2.resize(green_light, (100, 100))
yellow_light = cv2.resize(yellow_light, (100, 100))

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

def calculate_green_duration(vehicle_count):
    if vehicle_count > 10:
        additional_time = min((vehicle_count - 10) // 5 * 10, 40)
        return 20 + additional_time
    return 20

# Initialize traffic signal logic
traffic_signals = [red_light, red_light, red_light, red_light]
signal_timers = [0, 0, 0, 0]
green_durations = [20, 20, 20, 20]
yellow_duration = 3
last_green = -1
signal_change_time = time.time() + 3

# Define the class indices to detect
class_indices_to_detect = [classes.index(cls) for cls in classes_to_detect]

while True:
    # Process frames from all cameras
    frame1, count1 = process_frame(cap1, net, output_layers, class_indices_to_detect, 1/255.0, (512, 512), (0, 0, 0), True)
    frame2, count2 = process_frame(cap2, net, output_layers, class_indices_to_detect, 1/255.0, (512, 512), (0, 0, 0), True)
    frame3, count3 = process_frame(cap3, net, output_layers, class_indices_to_detect, 1/255.0, (512, 512), (0, 0, 0), True)
    frame4, count4 = process_frame(cap4, net, output_layers, class_indices_to_detect, 1/255.0, (512, 512), (0, 0, 0), True)

    current_time = time.time()

    # Update signal timers and traffic signals
    if current_time >= signal_change_time:
        vehicle_counts = [count1, count2, count3, count4]
        # Determine the next traffic light to be green
        next_green = np.argmax(vehicle_counts)

        # Ensure not the same signal consecutively
        if next_green == last_green:
            vehicle_counts[next_green] = -1  # Temporarily invalidate
            next_green = np.argmax(vehicle_counts)
        
        # Update the green light duration
        green_durations = [calculate_green_duration(count) for count in vehicle_counts]
        signal_change_time = current_time + green_durations[next_green] + yellow_duration
        last_green = next_green

        for i in range(4):
            traffic_signals[i] = green_light if i == next_green else red_light
            signal_timers[i] = signal_change_time - current_time

        # Adjust timers for subsequent signals
        for i in range(next_green + 1, next_green + 4):
            idx = i % 4
            signal_timers[idx] = signal_timers[next_green] + (i - next_green) * 20

    # Display logic
    top_row = np.hstack((frame1, frame2))
    bottom_row = np.hstack((frame3, frame4))
    split_screen = np.vstack((top_row, bottom_row))

    for i, (count, signal_timer) in enumerate(zip([count1, count2, count3, count4], signal_timers)):
        quadrant_x = (i % 2) * (window_width // 2)
        quadrant_y = (i // 2) * (window_height // 2)
        # Display traffic light
        split_screen[quadrant_y:quadrant_y + 100, quadrant_x:quadrant_x + 100] = traffic_signals[i]
        # Display vehicle count and timer
        cv2.putText(split_screen, f"Vehicles: {count}", (quadrant_x + window_width // 4 - 60, quadrant_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(split_screen, f"Timer: {int(signal_timer)}", (quadrant_x + window_width // 4 - 60, quadrant_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('Split Screen', split_screen)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
