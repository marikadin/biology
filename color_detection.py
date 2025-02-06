import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import mplcursors
import tkinter as tk
import re
from datetime import datetime

def extract_datetime_from_filename(filename):
    """Extract the datetime from the filename."""
    match = re.search(r'D\((\d{1,2})_(\d{1,2})_(\d{4})\)_T\((\d{1,2})_(\d{1,2})\)', filename)
    if match:
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        hour = int(match.group(4))
        minute = int(match.group(5))
        return datetime(year, month, day, hour, minute)
    return None

def process_image_folder(folder_path, target_color):
    time_diffs = []
    images = {}

    # Maximum possible difference between any two RGB colors (255, 255, 255) and (0, 0, 0)
    max_diff = np.linalg.norm(np.array([255, 255, 255]) - np.array([0, 0, 0]))

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Image {filename} not found.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            diff = np.linalg.norm(image_rgb - target_color, axis=2)
            min_idx = np.unravel_index(np.argmin(diff), diff.shape)
            min_color = image_rgb[min_idx]

            # Calculate the color difference (Euclidean distance)
            min_diff = np.min(diff)
            # Normalize the difference to a percentage (0-100%)
            min_diff_percent = (min_diff / max_diff) * 100

            time_diffs.append((filename, min_diff_percent, min_color, image_rgb))
            images[filename] = image_rgb  # Store full image for hovering

    # Sort by the datetime extracted from the filename
    time_diffs.sort(key=lambda x: extract_datetime_from_filename(x[0]))
    return time_diffs, images

def get_screen_size():
    """Get the screen width and height using tkinter."""
    root = tk.Tk()
    root.withdraw()  # Hide main tkinter window
    return root.winfo_screenwidth(), root.winfo_screenheight()

def plot_time_diffs(time_diffs, images):
    filenames, diffs, colors, _ = zip(*time_diffs)
    screen_width, screen_height = get_screen_size()
    diffs = list(diffs)  # Convert tuple to list
    for i in range(len(diffs)):
        diffs[i] = 100 - diffs[i]
        
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(filenames)), diffs, marker='o', linestyle='-')

    # Reduce x-axis labels
    step = max(1, len(filenames) // 10)  # Show at most 10 labels
    ax.set_xticks(range(0, len(filenames), step))
    ax.set_xticklabels(filenames[::step][15:23], rotation=90)

    ax.set_xlabel('Image Filename')
    ax.set_ylabel('Color Similarity (%)')
    ax.set_title('Color Difference Over Time')

    cursor = mplcursors.cursor(ax, hover=True)

    # Track previous figure to close before opening a new one
    previous_fig = [None]

    @cursor.connect("add")
    def on_hover(sel):
        idx = int(sel.index)
        filename = filenames[idx]
        color_diff = diffs[idx]
        color = colors[idx]
        image = images[filename]

        # Close previous figure if it exists
        if previous_fig[0] is not None:
            plt.close(previous_fig[0])

        # Create a small color patch
        color_patch = np.full((50, 50, 3), color, dtype=np.uint8)

        # Display the image and color patch in a new figure
        new_fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        previous_fig[0] = new_fig  # Store reference to close later

        axes[0].imshow(image)
        axes[0].axis("off")
        axes[0].set_title("Image")

        axes[1].imshow(color_patch)
        axes[1].axis("off")
        axes[1].set_title("Color")

        sel.annotation.set_text(f"{filename}\nSimilarity: {color_diff:.2f}%")

        # Move the new figure window to the bottom-right corner
        new_fig.canvas.manager.window.wm_geometry(f"+{screen_width-450}+{screen_height-250}")

        plt.show(block=False)

    plt.tight_layout()
    plt.show()

# Example usage
folder_path = r'crops\rect_1'
target_color = np.array([0, 100, 0])  # Example target color
time_diffs, images = process_image_folder(folder_path, target_color)
plot_time_diffs(time_diffs, images)
