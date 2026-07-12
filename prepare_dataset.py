import os
import random
import shutil

random.seed(42)

SOURCE_DIR = "dataset"
OUTPUT_DIR = "dataset_split"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

CLASSES = ["damaged", "intact"]
VIEWS = ["top", "side"]

# Create output folders
for split in ["train", "valid", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

for cls in CLASSES:

    images = []

    for view in VIEWS:

        folder = os.path.join(SOURCE_DIR, cls, view)

        for file in os.listdir(folder):

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                images.append((view, os.path.join(folder, file)))

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    valid_end = int(total * 0.85)

    train = images[:train_end]
    valid = images[train_end:valid_end]
    test = images[valid_end:]

    def copy_files(file_list, split):

        for i, (view, path) in enumerate(file_list):

            ext = os.path.splitext(path)[1]

            new_name = f"{view}_{i:04d}{ext}"

            shutil.copy(
                path,
                os.path.join(
                    OUTPUT_DIR,
                    split,
                    cls,
                    new_name
                )
            )

    copy_files(train, "train")
    copy_files(valid, "valid")
    copy_files(test, "test")

print("Dataset prepared successfully!")