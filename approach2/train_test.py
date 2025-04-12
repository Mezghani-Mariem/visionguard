import os
import shutil
import yaml
import cv2
from sklearn.model_selection import train_test_split

# Configuration
RAW_DATA_DIR = "raw_data"
OUTPUT_DIR = "yolo_dataset"
CLASSES = sorted(os.listdir(RAW_DATA_DIR))  # ['pedestrian', 'speed_limit', ...]

# Create YOLOv8 folder structure
os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/val", exist_ok=True)

# Process each class
for class_idx, class_name in enumerate(CLASSES):
    class_dir = f"{RAW_DATA_DIR}/{class_name}"
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    # Split into train/val (80/20)
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    # Helper function to process images
    def process_images(images, split='train'):
        for img in images:
            # Copy image
            shutil.copy(
                f"{class_dir}/{img}",
                f"{OUTPUT_DIR}/images/{split}/{class_name}_{img}"
            )
            
            # Create YOLO label file (one single centered bounding box)
            img_path = f"{OUTPUT_DIR}/images/{split}/{class_name}_{img}"
            h, w = cv2.imread(img_path).shape[:2]
            
            # Normalized center coordinates (assuming full-image bounding box)
            x_center, y_center = 0.5, 0.5
            width, height = 1.0, 1.0  # Covers entire image
            
            # Write label file
            label_path = f"{OUTPUT_DIR}/labels/{split}/{class_name}_{os.path.splitext(img)[0]}.txt"
            with open(label_path, 'w') as f:
                f.write(f"{class_idx} {x_center} {y_center} {width} {height}")

    # Process splits
    process_images(train_imgs, 'train')
    process_images(val_imgs, 'val')

# Generate dataset.yaml
data = {
    'path': os.path.abspath(OUTPUT_DIR),
    'train': 'images/train',
    'val': 'images/val',
    'names': {i: name for i, name in enumerate(CLASSES)}
}

with open(f"{OUTPUT_DIR}/dataset.yaml", 'w') as f:
    yaml.dump(data, f)

print(f"Dataset ready at {OUTPUT_DIR}/")
print(f"Classes: {CLASSES}")