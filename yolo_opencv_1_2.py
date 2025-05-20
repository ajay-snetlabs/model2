# export QT_QPA_PLATFORM=xcb
# python yolo_opencv.py --input video.mp4 --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt --confidence 0.4

# NEW Command
# python yolo_opencv.py --input video.mp4 --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt --confidence 0.4 --output output_video.avi


import cv2
import argparse
import numpy as np
import time
import os

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("argparse version:", argparse.__version__ if hasattr(argparse, '__version__') else "Part of standard library")

ap = argparse.ArgumentParser()
# Change '--image' to '--input' to handle both image and video
ap.add_argument('--input', required=True,
                help='path to input image or video file')
ap.add_argument('--config', required=True,
                help='path to yolo config file')
ap.add_argument('--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('--classes', required=True,
                help='path to text file containing class names')
ap.add_argument('--confidence', type=float, default=0.5,
                help='minimum probability to filter weak detections')
ap.add_argument('--output', help='path to output video file (optional)')
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

input_path = args.input
config_path = args.config
weights_path = args.weights
classes_path = args.classes
output_path = args.output

if not os.path.exists(input_path):
    print(f"Error: Input file not found at '{input_path}'")
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

classes = None
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(weights_path, config_path)

# Check if the input is an image or a video
if os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
    # Process image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not load the image at '{input_path}'")
        exit()
    Height, Width = image.shape[:2]
    scale = 0.00392
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
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
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
        x, y, w, h = round(box[0]), round(box[1]), round(box[2]), round(box[3])
        draw_prediction(image, class_ids[i], confidences[i], x, y, x + w, y + h)

    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
    resized_image = cv2.resize(image, (800, int(800 * Height / Width))) if Width > 800 else image
    cv2.imshow('Object Detection', resized_image)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    output_base, output_ext = os.path.splitext(os.path.basename(input_path))
    output_image_path = f"{output_base}_detected{output_ext}"
    cv2.imwrite(output_image_path, image)
    print(f"\n--- Image Analysis ---")
    print(f"Frame Size: {Width}x{Height}")
    print(f"Detection Time: {detection_time:.4f} seconds")
    if detection_time > 0:
        print(f"FPS: {1/detection_time:.2f} frames per second")
    print(f"Objects Detected: {len(indices)}")
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        print("\nConfidence Scores:")
        print(f"  Average: {avg_confidence*100:.2f}%")
        print(f"  Minimum: {min_confidence*100:.2f}%")
        print(f"  Maximum: {max_confidence*100:.2f}%")
    print(f"\nDetection result saved to: {output_image_path}")
    print("\nPress any key to close the window...")

elif os.path.isfile(input_path) and input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
    # Process video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{input_path}'")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = None  # Initialize out here
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            print(f"Error: Could not open output video file for writing at '{output_path}'")
            out = None

    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
    frame_count = 0
    total_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        Height, Width = frame.shape[:2]
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        start_time = time.time()
        outs = net.forward(get_output_layers(net))
        detection_time = time.time() - start_time
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = args.confidence
        nms_threshold = 0.4
        for out_layer in outs:
            for detection in out_layer:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
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
            x, y, w, h = round(box[0]), round(box[1]), round(box[2]), round(box[3])
            draw_prediction(frame, class_ids[i], confidences[i], x, y, x + w, y + h)

        cv2.imshow('Object Detection', frame)
        if out is not None:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    total_time = time.time() - total_start_time
    if frame_count > 0:
        avg_fps = frame_count / total_time
    else:
        avg_fps = 0

    print(f"\n--- Video Analysis ---")
    print(f"Frame Size: {frame_width}x{frame_height}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total Frames Processed: {frame_count}")
    # Note: Getting accurate per-frame object count and confidence scores for the entire video
    # without storing all of them would require significant modification.
    # The following prints information from the last processed frame.
    print(f"Objects Detected (last frame): {len(indices)}")
    if confidences:
        avg_confidence_last = sum(confidences) / len(confidences)
        min_confidence_last = min(confidences)
        max_confidence_last = max(confidences)
        print("\nConfidence Scores (last frame):")
        print(f"  Average: {avg_confidence_last*100:.2f}%")
        print(f"  Minimum: {min_confidence_last*100:.2f}%")
        print(f"  Maximum: {max_confidence_last*100:.2f}%")

else:
    print(f"Error: Unsupported input file format: '{input_path}'. Please provide a valid image or video file.")
#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################