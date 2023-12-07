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

# Open four video capture objects
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

# Initialize variables for snapshot
snapshot_time = None
snapshot_count = 0

while True:
    # Read frames from all four videos
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    if not (ret1 and ret2 and ret3 and ret4):
        # If any of the videos end, break the loop
        break

    # Resize frames to fit the window
    frame1 = cv2.resize(frame1, (window_width // 2, window_height // 2))
    frame2 = cv2.resize(frame2, (window_width // 2, window_height // 2))
    frame3 = cv2.resize(frame3, (window_width // 2, window_height // 2))
    frame4 = cv2.resize(frame4, (window_width // 2, window_height // 2))

    # Arrange frames in the split-screen layout
    top_row = np.hstack((frame1, frame2))
    bottom_row = np.hstack((frame3, frame4))
    split_screen = np.vstack((top_row, bottom_row))

    # Detecting specific classes in each frame
    frames = [frame1, frame2, frame3, frame4]

    for idx, frame in enumerate(frames):
        height, width, channels = frame.shape

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
                if confidence > 0.1 and class_id in class_indices_to_detect:
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

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)

        # Draw bounding boxes without labels on the frame
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Place the frame in the corresponding quadrant of the split-screen
        if idx == 0:
            split_screen[:window_height // 2, :window_width // 2] = frame
        elif idx == 1:
            split_screen[:window_height // 2, window_width // 2:] = frame
        elif idx == 2:
            split_screen[window_height // 2:, :window_width // 2] = frame
        elif idx == 3:
            split_screen[window_height // 2:, window_width // 2:] = frame

    # Show the split-screen output
    cv2.imshow("Split Screen", split_screen)

    # Check for snapshot time
    if snapshot_time is None:
        snapshot_time = time.time()
    elif time.time() - snapshot_time >= 2:
        # Capture snapshots after 10 seconds
        snapshot_count += 1
        for i, frame in enumerate(frames):
            cv2.imwrite(f'snapshot{snapshot_count}_camera{i+1}.jpg', frame)
        snapshot_time = None

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