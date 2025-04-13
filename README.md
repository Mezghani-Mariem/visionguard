# visionguard
VisionGuard is a computer vision project focused on traffic sign detection and classification, utilizing both traditional CNN architectures and cutting-edge object detection YOLO model.
We used two different approaches in this project.


Second approach(transfer learning ) — using a YOLOv8n model:advanced CNN that uses convolutional layers
We trained a YOLOv8n model on a dataset we created using a Python script.
This project was done in three steps:
Step 1: Creating the dataset: 5 classes, ~1250 images each.
Classes are ('travaux', 'warning', 'dodane', 'left', 'ceder_le_passage').
The Python script that creates the dataset detects triangle shapes, crops the image, and saves it to a folder, creating the dataset (all the signs we used have triangular shapes).
Step 2: Generating labels and training the model.
Step 3: Using the model to detect traffic signs.
