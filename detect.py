# In order to get configFile use the below command
# python tf_text_graph_ssd.py --input models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --config pipeline.config --output /ssd_mobilenet_v2_coco_2018_03_29.pbtxt

import cv2
import sys

modelFile = "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
configFile = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classFile = "coco_class_labels.txt"
s = 0

# Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7

# Get list of object labels
with open(classFile) as fp:
    labels = fp.read().split("\n")

if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)
win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Read the Tensorflow network
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame,1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB = False, crop = False)
    # Run a model
    net.setInput(blob)
    detections = net.forward()

    # For every Detected Object
    for i in range(detections.shape[2]):
        # Find the class and confidence 
        confidence = detections[0, 0, i, 2]
        classId = int(detections[0, 0, i, 1])

        # Check if the detection is of good quality
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            # Frame around the detected object
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0),2)
            label = "{}".format(labels[classId])
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Add label to object
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]),
                                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                                (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)
