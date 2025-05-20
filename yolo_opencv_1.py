# export QT_QPA_PLATFORM=xcb
# python yolo_opencv.py --image dog.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt --confidence 0.4

# NEW Command
# python yolo_opencv_1.py --image traffic.mp4 --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt --confidence 0.6


import cv2
import argparse
import numpy as np
import time
import os

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("argparse version:", argparse.__version__ if hasattr(argparse, '__version__') else "Part of standard library")


ap = argparse.ArgumentParser()
ap.add_argument('--image', required=True,
                help='path to input image')
ap.add_argument('--config', required=True,
                help='path to yolo config file')
ap.add_argument('--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('--classes', required=True,
                help='path to text file containing class names')
# Use '--confidence' to match the command
ap.add_argument('--confidence', type=float, default=0.5,
                help='minimum probability to filter weak detections')
args = ap.parse_args()


def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{classes[class_id]}: {confidence:.2f}"
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


image_path = args.image
config_path = args.config
weights_path = args.weights
classes_path = args.classes

if not os.path.exists(image_path):
    print(f"Error: Image file not found at '{image_path}'")
    exit()
if not os.path.exists(config_path):
    print(f"Error: Configuration file not found at '{config_path}'")
    exit()
if not os.path.exists(weights_path):
    print(f"Error: Weights file not found at '{weights_path}'")
    exit()
if not os.path.exists(classes_path):
    print(f"Error: Classes file not found at '{classes_path}'")
    exit()

image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load the image at '{image_path}'")
    exit()

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(weights_path, config_path)
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
start_time = time.time()
outs = net.forward(get_output_layers(net))
detection_time = time.time() - start_time

class_ids = []
confidences = []
boxes = []
conf_threshold = args.confidence
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
    try:
        box = boxes[i]
    except:
        i = i[0]
        box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
height, width = image.shape[:2]
max_height = 800
if height > max_height:
    ratio = max_height / height
    width = int(width * ratio)
    height = max_height
    image = cv2.resize(image, (width, height))
cv2.imshow('Object Detection', image)
print("Press any key to close the window...")
cv2.waitKey(0)
cv2.destroyAllWindows()

output_path = "object-detection.jpg"
cv2.imwrite(output_path, image)

print("\n--- Detection Metrics ---")
print(f"Detection time: {detection_time:.4f} seconds")
print(f"FPS: {1/detection_time:.2f} frames per second")
print(f"Objects detected: {len(indices)}")

if confidences:
    avg_confidence = sum(confidences) / len(confidences)
    min_confidence = min(confidences)
    max_confidence = max(confidences)
    print("\nConfidence Scores:")
    print(f"  Average: {avg_confidence*100:.2f}%")
    print(f"  Minimum: {min_confidence*100:.2f}%")
    print(f"  Maximum: {max_confidence*100:.2f}%")

print(f"\nDetection result saved to: {output_path}")
print("\nPress any key to close the window...")


#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################