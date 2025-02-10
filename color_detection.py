import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import mplcursors
import tkinter as tk
import re
from datetime import datetime
from matplotlib.widgets import CheckButtons

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

def plot_all_time_diffs(all_time_diffs, all_images, folder_names):
    screen_width, screen_height = get_screen_size()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Color map for the different folders
    colors_list = plt.cm.get_cmap('tab10', len(folder_names)).colors

    # Store line plots for toggling visibility
    lines = []
    regression_lines = []

    for idx, (time_diffs, folder_name) in enumerate(zip(all_time_diffs, folder_names)):
        filenames, diffs, colors, _ = zip(*time_diffs)
        dates = [extract_datetime_from_filename(filename) for filename in filenames]
        dates = [date.strftime("%Y-%m-%d %H:%M") for date in dates]
        diffs = list(diffs)

        for i in range(len(diffs)):
            diffs[i] = 100 - diffs[i]

        x_values = np.arange(len(filenames))  # X-axis values (indices)
        y_values = np.array(diffs)  # Y-axis values (similarity %)

        # Plot the main line
        line, = ax.plot(x_values, y_values, marker='o', linestyle='-', color=colors_list[idx])
        lines.append(line)

        # Compute and plot the linear regression line
        if len(x_values) > 1:  # Avoid issues with single data points
            coeffs = np.polyfit(x_values, y_values, 1)  # Linear regression (degree 1)
            regression_fn = np.poly1d(coeffs)
            reg_line, = ax.plot(x_values, regression_fn(x_values), linestyle='--', color=colors_list[idx], alpha=0.7)
            regression_lines.append(reg_line)

    # Reduce x-axis labels
    step = max(1, len(filenames) // 10)  # Show at most 10 labels
    ax.set_xticks(range(0, len(filenames), step))
    ax.set_xticklabels(dates[::step], rotation=45)

    ax.set_xlabel('Time')
    ax.set_ylabel('Color Similarity (%)')
    ax.set_title('Color Difference Over Time for Selected Folders')

    cursor = mplcursors.cursor(ax, hover=True)

    # Track previous figure to close before opening a new one
    previous_fig = [None]

    @cursor.connect("add")
    def on_hover(sel):
        idx = int(sel.index)
        filename = filenames[idx]
        color_diff = diffs[idx]
        color = colors[idx]
        image = all_images[filename]

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

        sel.annotation.set_text(f"{filename}\nSimilarity: {color_diff:.2f}%\nRGB: {int(color[0]),int(color[1]),int(color[2])}")

        # Move the new figure window to the bottom-right corner
        new_fig.canvas.manager.window.wm_geometry(f"+{screen_width-450}+{screen_height-250}")

        plt.show(block=False)

    plt.tight_layout()
    plt.legend()

    rax = plt.axes([0.933, 0.775, 0.052, 0.15])
    check = CheckButtons(rax, folder_names, [True] * len(folder_names))

    # Set the label colors to match the folder colors
    [label.set_color(color) for label, color in zip(check.labels, colors_list)]

    def toggle_line(label):
        index = folder_names.index(label)
        visible = not lines[index].get_visible()
        lines[index].set_visible(visible)
        regression_lines[index].set_visible(visible)  # Toggle regression line too
        plt.draw()

    check.on_clicked(toggle_line)

    plt.show()

def ask_user_for_folder_selection(parent_folder):
    """Ask the user to select which subfolders inside a parent folder to display."""
    subfolders = [f.name for f in os.scandir(parent_folder) if f.is_dir()]
    print("Available subfolders:")
    for idx, subfolder in enumerate(subfolders, 1):
        print(f"{idx}. {subfolder}")

    selected_indexes = input("Enter the numbers of the subfolders to display (comma-separated): ").split(',')
    selected_folders = [subfolders[int(idx) - 1] for idx in selected_indexes]

    return selected_folders

def plot_selected_folders(selected_folders, parent_folder, target_color):
    """Process and plot images from selected subfolders."""
    all_time_diffs = []
    all_images = {}

    for folder in selected_folders:
        folder_path = os.path.join(parent_folder, folder)
        time_diffs, images = process_image_folder(folder_path, target_color)
        all_time_diffs.append(time_diffs)
        all_images.update(images)

    # Plot all selected folders on the same plot
    plot_all_time_diffs(all_time_diffs, all_images, selected_folders)

# Example usage
parent_folder = 'crops'
target_color = np.array([0, 50, 0])  # Example target color

# Ask the user to select subfolders
selected_folders = ask_user_for_folder_selection(parent_folder)

# Plot the selected folders
plot_selected_folders(selected_folders, parent_folder, target_color)
