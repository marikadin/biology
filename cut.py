import os
import cv2
import numpy as np


def remove_black_images(folder, threshold=10):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        avg_intensity = np.mean(img)
        if avg_intensity < threshold:
            os.remove(img_path)
            print(f"Removed {filename} (avg intensity: {avg_intensity})")


def select_rectangle(event, x, y, flags, param):
    global drawing, start_x, start_y, rects, img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Create a copy of the image to draw the rectangle live
            img_copy = img.copy()
            cv2.rectangle(img_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("Draw Rectangles and Press 'q' to Save", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        rects.append((start_x, start_y, end_x, end_y))
        print(f"Rectangle selected: {start_x}, {start_y}, {end_x}, {end_y}")
        # Draw the final rectangle and show it
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.imshow("Draw Rectangles and Press 'q' to Save", img)


def crop_and_save(image, rects, output_folder, image_name):
    # Save the cropped parts into separate folders for each rectangle
    for i, (x1, y1, x2, y2) in enumerate(rects):
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Ensure valid cropping coordinates
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > image.shape[1]: x2 = image.shape[1]
        if y2 > image.shape[0]: y2 = image.shape[0]

        # Check if the crop has a valid size
        if x2 > x1 and y2 > y1:
            crop = image[y1:y2, x1:x2]

            # Create a subfolder for each rectangle and save the crops
            subfolder = os.path.join(output_folder, f"rect_{i + 1}")
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)

            save_path = os.path.join(subfolder, f"{image_name}_crop_{i + 1}.png")

            # Ensure the crop is not empty before saving
            if crop.size > 0:
                cv2.imwrite(save_path, crop)
                print(f"Saved: {save_path}")
            else:
                print(f"Skipping empty crop for {image_name} at rect {i + 1}")
        else:
            print(f"Invalid crop size for {image_name} at rect {i + 1}. Skipping...")


def main():
    folder = "camera"  # Change to your folder path
    remove_black_images(folder)

    images = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images left in folder!")
        return

    global drawing, start_x, start_y, rects, img
    drawing = False
    start_x = start_y = 0
    rects = []

    # Step 1: Choose rectangles on the first image
    first_image_path = os.path.join(folder, images[0])
    img = cv2.imread(first_image_path)

    cv2.imshow("Draw Rectangles and Press 'q' to Save", img)
    cv2.setMouseCallback("Draw Rectangles and Press 'q' to Save", select_rectangle)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # Step 2: Apply the selected rectangles to all other images
    for image_name in images:
        img_path = os.path.join(folder, image_name)
        img = cv2.imread(img_path)

        # Step 3: Save the cropped parts for each image
        crop_and_save(img, rects, "crops", image_name)


if __name__ == "__main__":
    main()
