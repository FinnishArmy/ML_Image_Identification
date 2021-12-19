import cv2

# Set the image to use.
# Change the 'index.jpg' to image name if you change the image.
image_test = cv2.imread('index.jpg')

# This was coded with the help of this video
# https://www.youtube.com/watch?v=IBQYqwq_w14

# Create an matrix for the names.
classNames = ['person', 'bicycle', 'car', 'motorcycle',
              'airplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'street sign',
              'stop sign', 'parking meter', 'bench', 'bird',
              'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
              'bear', 'zebra', 'giraffe', 'hat', 'backpack',
              'umbrella', 'shoe', 'eye glasses', 'handbag',
              'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat',
              'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'plate', 'wine glass',
              'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
              'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'mirror', 'dining table',
              'window', 'desk', 'toilet', 'door', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'blender',
              'book', 'clock', 'vase', 'scissors', 'teddy bear',
              'hair drier', 'toothbrush', 'hair brush']

classColours = [(204,78,210), (0,192,255), (0,131,0), (240,176,0),
                (254,100,38), (0,0,255), (182,117,46), (185,60,129),
                (204,153,255), (80,208,146), (0,0,204), (17,90,197),
                (0,255,255), (102,255,102), (255,255,0), (0, 255, 0),
                (0, 0, 255), (255, 0, 0), (204,78,210), (0,192,255),
                (0,131,0), (240,176,0), (254,100,38), (0,0,255),
                (182,117,46), (185,60,129), (204,153,255), (80,208,146),
                (0,0,204), (17,90,197), (0,255,255), (102,255,102),
                (255,255,0), (204,78,210), (0,192,255), (0,131,0),
                (240,176,0), (254,100,38), (0,0,255), (182,117,46),
                (185,60,129), (204,153,255), (80,208,146), (0,0,204), (17,90,197),
                (0,255,255), (102,255,102), (255,255,0)]

# Import the needed files.
indexes = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights = 'frozen_inference_graph.pb'

# Build the model using OpenCV
web = cv2.dnn_DetectionModel(weights, indexes)

# Default configs
web.setInputSize(320, 320)
web.setInputScale(1.0 / 127.5)
web.setInputMean((127.5, 127.5, 127.5))
web.setInputSwapRB(True)

# Get the bounding box and set a threshold of 50% confidence.
# Change confThreshold to change how confident the model is.
class_indexes, confidence_level, bounding_box = web.detect(image_test, confThreshold=0.5)

# Draw the rectangles for each object
for class_index, confidence_level, box in zip(class_indexes.flatten(), confidence_level.flatten(), bounding_box):
    # Create a rectangle
    cv2.rectangle(image_test, box, color = classColours[class_index], thickness=2)
    # Apply the text above the rectangle
    cv2.putText(image_test, classNames[class_index-1], (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

# Show the image in a window
cv2.imshow("Output", image_test)
cv2.waitKey(0)

