import cv2
import numpy as np
import time

# Load standard YOLOv4 model
net_standard = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names_standard = net_standard.getLayerNames()
output_layers_standard = [layer_names_standard[i - 1] for i in net_standard.getUnconnectedOutLayers().flatten()]

# Load custom YOLOv4 model for vehicle orientation
net_custom = cv2.dnn.readNet("yolov4_vehicle_orientation_74500.weights", "yolov4_vehicle_orientation.cfg")
layer_names_custom = net_custom.getLayerNames()
output_layers_custom = [layer_names_custom[i - 1] for i in net_custom.getUnconnectedOutLayers().flatten()]

# Load labels from vehicle_orientation.names
with open("vehicle_orientation.names", "r") as f:
    classes_custom = [line.strip() for line in f.readlines()]

# Define the class names to filter out (back and side of vehicles)
classes_to_filter = ["car_back", "car_side", "bus_back", "bus_side", "truck_back", "truck_side", "motorcycle_back", "motorcycle_side", "cycle_back", "cycle_side"]
filter_indices = [classes_custom.index(cls) for cls in classes_to_filter if cls in classes_custom]

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

def process_frame(cap, net_standard, output_layers_standard, net_custom, output_layers_custom, filter_indices, scale_factor, size, mean, swapRB):
    ret, frame = cap.read()
    if not ret:
        print("No frame captured from the video stream")
        return None
    frame = cv2.resize(frame, (window_width // 2, window_height // 2))
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, scale_factor, size, mean, swapRB, crop=False)
    net_standard.setInput(blob)
    outs = net_standard.forward(output_layers_standard)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Pass the detected object through the custom model
                blob_custom = cv2.dnn.blobFromImage(frame[y:y+h, x:x+w], scale_factor, size, mean, swapRB, crop=False)
                net_custom.setInput(blob_custom)
                outs_custom = net_custom.forward(output_layers_custom)

                for out_custom in outs_custom:
                    for detection_custom in out_custom:
                        scores_custom = detection_custom[5:]
                        class_id_custom = np.argmax(scores_custom)
                        confidence_custom = scores_custom[class_id_custom]
                        if confidence_custom > 0.3 and class_id_custom not in filter_indices:
                            # Object is not a back or side orientation, add to detection
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            break

    return frame

# Timing mechanism for snapshot every 5 seconds
last_time = time.time()
interval = 5  # 5 seconds

while True:
    current_time = time.time()
    if current_time - last_time >= interval:
        print("Processing new set of frames")
        frame1 = process_frame(cap1, net_standard, output_layers_standard, net_custom, output_layers_custom, filter_indices, 1/255.0, (512, 512), (0, 0, 0), True)
        frame2 = process_frame(cap2, net_standard, output_layers_standard, net_custom, output_layers_custom, filter_indices, 1/255.0, (512, 512), (0, 0, 0), True)
        frame3 = process_frame(cap3, net_standard, output_layers_standard, net_custom, output_layers_custom, filter_indices, 1/255.0, (512, 512), (0, 0, 0), True)
        frame4 = process_frame(cap4, net_standard, output_layers_standard, net_custom, output_layers_custom, filter_indices, 1/255.0, (512, 512), (0, 0, 0), True)
        last_time = current_time

        # Check if any frame is None
        if None in [frame1, frame2, frame3, frame4]:
            print("One or more frames are None, skipping this iteration")
            continue

        # Display logic
        top_row = np.hstack((frame1, frame2))
        bottom_row = np.hstack((frame3, frame4))
        split_screen = np.vstack((top_row, bottom_row))
        
        cv2.imshow('Split Screen', split_screen)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break
        break

cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
