#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import argparse
import numpy as np
import time

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("argparse version:", argparse.__version__ if hasattr(argparse, '__version__') else "Part of standard library")


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)

# Start timing the detection
start_time = time.time()

# Run detection
outs = net.forward(get_output_layers(net))

# Calculate detection time
detection_time = time.time() - start_time

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4


for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
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
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

# Check if image was loaded properly
if image is None:
    print("Error: Could not load the image")
    exit(1)

# Create a resizable window
cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)

# Resize the window to fit the screen (optional)
height, width = image.shape[:2]
max_height = 800
if height > max_height:
    ratio = max_height / height
    width = int(width * ratio)
    height = max_height
    image = cv2.resize(image, (width, height))

# Show the image
cv2.imshow('Object Detection', image)
print("Press any key to close the window...")

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
output_path = "object-detection.jpg"
cv2.imwrite(output_path, image)

# Calculate and print metrics
print("\n--- Detection Metrics ---")
print(f"Detection time: {detection_time:.4f} seconds")
print(f"FPS: {1/detection_time:.2f} frames per second")
print(f"Objects detected: {len(indices)}")

# Calculate confidence statistics
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


# export QT_QPA_PLATFORM=xcb
# python yolo_opencv.py --image dog.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt