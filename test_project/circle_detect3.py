import cv2
import numpy as np

# Load YOLOv4 weights and configuration
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# Load classes
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set input size
net_width, net_height = 608, 608

# Create blob from input image
def create_blob(image):
    blob = cv2.dnn.blobFromImage(image, 1/255, (net_width, net_height), [0, 0, 0], swapRB=True, crop=False)
    return blob

# Define minimum confidence threshold
conf_threshold = 0.5

# Define NMS threshold
nms_threshold = 0.4

# Load video file
cap = cv2.VideoCapture('/Users/kimjunho/Desktop/OpenCV_study/videos/circle_detect.mp4')

while(cap.isOpened()):
    # Read frame from video
    ret, frame = cap.read()

    # Create blob from input frame
    blob = create_blob(frame)

    # Set input to network
    net.setInput(blob)

    # Forward pass
    output_layers = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers)

    # Post-process detections
    boxes = []
    confidences = []
    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == 0:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width/2)
                top = int(center_y - height/2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw boxes on the frame
    for i in indices:
        i = i[0]
        box = boxes[i]
        left, top, width, height = box
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
