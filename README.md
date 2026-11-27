# EXP-12 -PROJECT: Face Detection with Haar Cascades

## Aim

To write Python programs using OpenCV to perform the following image processing operations:

i) Perform ROI Segmentation in an Image using Bitwise AND.

ii) Perform Handwriting Detection in an Image.

iii) Perform Object Detection with Labels in an Image using MobileNet-SSD.

## Software Required

- Anaconda - Python 3.7 or above
- OpenCV library (opencv-python)
- NumPy library (numpy)
- Matplotlib library (matplotlib)
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) ROI Segmentation in an Image using Bitwise AND

Step 1: Import the necessary packages: cv2, numpy and matplotlib.pyplot.

Step 2: Read the input image using cv2.imread().

Step 3: Convert the image from BGR to RGB format.

Step 4: Display the original image using plt.imshow().

Step 5: Define the Region of Interest (ROI) using pixel coordinates.

Step 6: Create a blank mask using np.zeros_like().

Step 7: Place the selected ROI on the mask.

Step 8: Perform the Bitwise AND operation using cv2.bitwise_and().

Step 9: Convert the segmented image into RGB format.

Step 10: Display the segmented ROI using Matplotlib.

### II) Handwriting Detection in an Image

Step 1: Import the necessary packages: cv2, numpy and matplotlib.pyplot.

Step 2: Read the input image using cv2.imread().

Step 3: Convert the image from BGR to RGB format and display the original image.

Step 4: Convert the image into grayscale using cv2.cvtColor().

Step 5: Apply Gaussian Blur using cv2.GaussianBlur() to reduce noise.

Step 6: Apply Canny Edge Detection using cv2.Canny().

Step 7: Find contours using cv2.findContours().

Step 8: Filter contours based on their area.

Step 9: Calculate bounding boxes using cv2.boundingRect().

Step 10: Draw rectangles around the detected regions.

Step 11: Display the final handwriting detection result using Matplotlib.

### III) Object Detection with Labels in an Image using MobileNet-SSD

Step 1: Import the necessary packages: cv2, numpy and matplotlib.pyplot.

Step 2: Load the pretrained MobileNet-SSD model using the configuration file and weights file.

Step 3: Define the class labels for the objects.

Step 4: Read the input image using cv2.imread().

Step 5: Convert the image from BGR to RGB format.

Step 6: Create a blob using cv2.dnn.blobFromImage().

Step 7: Set the blob as input to the neural network using net.setInput().

Step 8: Perform object detection using net.forward().

Step 9: Extract the confidence score and class index from the detections.

Step 10: Filter detected objects using a confidence threshold of 0.5.

Step 11: Calculate bounding box coordinates.

Step 12: Draw rectangles and object labels using cv2.rectangle() and cv2.putText().

Step 13: Display the final object detection output using Matplotlib.

## Program

### I) ROI Segmentation in an Image using Bitwise AND

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('parrot.webp')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('on')
plt.show()

roi = image[100:420, 200:550]

mask = np.zeros_like(image)
mask[100:420, 200:550] = roi

segmented_roi = cv2.bitwise_and(image, mask)

segmented_roi_rgb = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2RGB)

plt.imshow(segmented_roi_rgb)
plt.title("Segmented ROI")
plt.axis('off')
plt.show()
```

### II) Handwriting Detection in an Image

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('note.jpg')

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

edges = cv2.Canny(blurred_image, 50, 150)

plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
plt.show()

contours, _ = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

result_image = image.copy()

for contour in contours:
    if cv2.contourArea(contour) > 50:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title("Handwriting Detection")
plt.axis('off')
plt.show()
```

### III) Object Detection with Labels in an Image using MobileNet-SSD

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

config_file = 'deploy.prototxt'
weights = 'mobilenet_iter_73000.caffemodel'

net = cv2.dnn.readNetFromCaffe(config_file, weights)
print("Model loaded successfully")

class_labels = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

print("Class labels created successfully")

image = cv2.imread('download.webp')

(h, w) = image.shape[:2]

print("Image loaded successfully")
print("Height:", h)
print("Width:", w)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

blob = cv2.dnn.blobFromImage(
    image,
    0.007843,
    (300, 300),
    127.5
)

print("Blob created successfully")

net.setInput(blob)

detections = net.forward()

print("Object detection completed successfully")

for i in range(detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:

        index = int(detections[0, 0, i, 1])
        label = class_labels[index]

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        (startX, startY, endX, endY) = box.astype("int")

        cv2.rectangle(image_rgb, (startX, startY), (endX, endY), (0, 255, 0), 2)

        cv2.putText(image_rgb, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

print("Rectangles and labels added successfully")

plt.figure(figsize=(10, 8))
plt.imshow(image_rgb)
plt.title("Object Detection with MobileNet-SSD")
plt.axis("off")
plt.show()

```

## Output

### I) ROI Segmentation

### Original image 


<img width="566" height="397" alt="image" src="https://github.com/user-attachments/assets/f850b562-446a-4598-a470-12ecf6e77610" />


### Segmented ROI image are displayed.


<img width="495" height="360" alt="image" src="https://github.com/user-attachments/assets/41282a22-27e8-44d8-89d1-2a6279d04ad0" />


### II) Handwriting Detection

### Original image


<img width="506" height="329" alt="image" src="https://github.com/user-attachments/assets/edb4eec9-c539-4482-9de7-23b56586883c" />


### Canny Edge Detection output 


<img width="508" height="303" alt="image" src="https://github.com/user-attachments/assets/cf0484c3-cbc3-4342-9b6e-4c7a815ebf73" />


### Handwriting detection result with bounding boxes are displayed.


<img width="506" height="311" alt="image" src="https://github.com/user-attachments/assets/2f181784-d5c4-43ae-8a6a-5a27d0756bff" />


### III) Object Detection using MobileNet-SSD

### Detected objects are displayed with green bounding boxes and their corresponding class labels.


<img width="791" height="557" alt="image" src="https://github.com/user-attachments/assets/da868227-db2d-4581-ae46-50d12abbec1a" />


## Result

Thus, the programs for ROI Segmentation using Bitwise AND, Handwriting Detection and Object Detection using MobileNet-SSD were successfully executed using OpenCV.

