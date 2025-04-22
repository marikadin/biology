import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import mplcursors
import tkinter as tk
import re
from datetime import datetime
from matplotlib.widgets import CheckButtons
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod

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

def create_sequences(data, seq_length):
    """Create sequences for model training"""
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:(i + seq_length)]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

class TimeSeriesModel(ABC):
    """Abstract base class for time series prediction models"""
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass

class LSTMPredictor(TimeSeriesModel):
    def __init__(self, seq_length=3):
        self.seq_length = seq_length
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.scaler = MinMaxScaler()
        
    def train(self, X, y):
        X = X.reshape((X.shape[0], X.shape[1], 1))
        self.model.fit(X, y, epochs=100, verbose=0)
    
    def predict(self, X):
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return self.model.predict(X, verbose=0)

class LinearRegressionPredictor(TimeSeriesModel):
    def __init__(self, seq_length=3):
        self.seq_length = seq_length
        self.model = LinearRegression()
        
    def train(self, X, y):
        X = X.reshape((X.shape[0], X.shape[1]))
        self.model.fit(X, y)
    
    def predict(self, X):
        X = X.reshape((X.shape[0], X.shape[1]))
        return self.model.predict(X).reshape(-1, 1)

def predict_future_values(data, model_type='lstm', n_future=5, seq_length=3):
    """
    Predict future values using specified model
    Args:
        data: Input time series data
        model_type: 'lstm' or 'linear'
        n_future: Number of future points to predict
        seq_length: Length of input sequences
    """
    if len(data) <= seq_length:
        return None, None, None
    
    # Initialize model based on type
    if model_type.lower() == 'lstm':
        model = LSTMPredictor(seq_length)
    elif model_type.lower() == 'linear':
        model = LinearRegressionPredictor(seq_length)
    else:
        raise ValueError("Model type must be 'lstm' or 'linear'")
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create sequences for training
    X, y = create_sequences(scaled_data, seq_length)
    
    # Train the model
    model.train(X, y)
    
    # Generate predictions for existing data
    existing_predictions = []
    for i in range(seq_length, len(scaled_data)):
        seq = scaled_data[i-seq_length:i].reshape(1, seq_length)
        pred = model.predict(seq)
        existing_predictions.append(pred[0, 0])
    
    # Predict future values
    future_predictions = []
    last_sequence = scaled_data[-seq_length:]
    
    for _ in range(n_future):
        next_pred = model.predict(last_sequence.reshape(1, seq_length))
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.vstack([last_sequence[1:], next_pred])
    
    # Inverse transform predictions
    existing_predictions = scaler.inverse_transform(np.array(existing_predictions).reshape(-1, 1))
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    # Calculate confidence intervals
    pred_std = np.std(existing_predictions - data[seq_length:])
    confidence = 1.96 * pred_std  # 95% confidence interval
    
    return existing_predictions, future_predictions, confidence

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

            min_diff = np.min(diff)
            min_diff_percent = (min_diff / max_diff) * 100

            time_diffs.append((filename, min_diff_percent, min_color, image_rgb))
            images[filename] = image_rgb

    time_diffs.sort(key=lambda x: extract_datetime_from_filename(x[0]))
    return time_diffs, images

def get_screen_size():
    """Get the screen width and height using tkinter."""
    root = tk.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()

def plot_all_time_diffs(all_time_diffs, all_images, folder_names, model_type='lstm'):
    screen_width, screen_height = get_screen_size()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_list = plt.cm.get_cmap('tab10', len(folder_names)).colors
    lines = []
    prediction_lines = []
    future_lines = []

    for idx, (time_diffs, folder_name) in enumerate(zip(all_time_diffs, folder_names)):
        filenames, diffs, colors, _ = zip(*time_diffs)
        dates = [extract_datetime_from_filename(filename) for filename in filenames]
        dates = [date.strftime("%Y-%m-%d %H:%M") for date in dates]
        diffs = list(diffs)

        for i in range(len(diffs)):
            diffs[i] = 100 - diffs[i]

        x_values = np.arange(len(filenames))
        y_values = np.array(diffs)

        # Plot the main line
        line, = ax.plot(x_values, y_values, marker='o', linestyle='-', color=colors_list[idx])
        lines.append(line)

        # Compute and plot predictions
        if len(y_values) > 3:
            existing_pred, future_pred, confidence = predict_future_values(y_values, model_type=model_type)
            
            if existing_pred is not None:
                # Plot predictions for existing data
                pred_x = np.arange(3, len(filenames))
                pred_line, = ax.plot(pred_x, existing_pred, 
                                   linestyle='--', color=colors_list[idx], alpha=0.7,
                                   label=f'{folder_name} {model_type.upper()}')
                prediction_lines.append(pred_line)
                
                # Plot future predictions
                future_x = np.arange(len(filenames), len(filenames) + 5)
                future_line, = ax.plot(future_x, future_pred, 
                                     linestyle=':', color=colors_list[idx], alpha=0.5)
                future_lines.append(future_line)
                
                # Add prediction confidence interval
                ax.fill_between(future_x, 
                              future_pred.flatten() - confidence,
                              future_pred.flatten() + confidence,
                              color=colors_list[idx], alpha=0.1)
                
                # Add text annotation for prediction trend
                future_trend = (future_pred[-1] - future_pred[0])[0]
                direction = "increasing" if future_trend > 0 else "decreasing"
                ax.text(0.02, 0.98 - idx*0.05, 
                       f"{folder_name}: {model_type.upper()} predicts {direction} trend",
                       transform=ax.transAxes, fontsize=8, color=colors_list[idx])

    # Reduce x-axis labels
    step = max(1, len(filenames) // 10)
    ax.set_xticks(range(0, len(filenames), step))
    ax.set_xticklabels(dates[::step], rotation=45)

    ax.set_xlabel('Time')
    ax.set_ylabel('Color Similarity (%)')
    ax.set_title(f'Color Graph With {model_type.upper()} Predictions')

    cursor = mplcursors.cursor(ax, hover=True)
    previous_fig = [None]

    @cursor.connect("add")
    def on_hover(sel):
        idx = int(sel.index)
        filename = filenames[idx]
        color_diff = diffs[idx]
        color = colors[idx]
        image = all_images[filename]

        if previous_fig[0] is not None:
            plt.close(previous_fig[0])

        color_patch = np.full((50, 50, 3), color, dtype=np.uint8)

        new_fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        previous_fig[0] = new_fig

        axes[0].imshow(image)
        axes[0].axis("off")
        axes[0].set_title("Image")

        axes[1].imshow(color_patch)
        axes[1].axis("off")
        axes[1].set_title("Color")

        sel.annotation.set_text(f"{filename}\nSimilarity: {color_diff:.2f}%\nRGB: {int(color[0]),int(color[1]),int(color[2])}")

        new_fig.canvas.manager.window.wm_geometry(f"+{screen_width-450}+{screen_height-250}")
        plt.show(block=False)

    plt.tight_layout()
    plt.legend()

    rax = plt.axes([0.933, 0.775, 0.052, 0.15])
    check = CheckButtons(rax, folder_names, [True] * len(folder_names))

    [label.set_color(color) for label, color in zip(check.labels, colors_list)]

    def toggle_line(label):
        index = folder_names.index(label)
        visible = not lines[index].get_visible()
        lines[index].set_visible(visible)
        if index < len(prediction_lines):
            prediction_lines[index].set_visible(visible)
        if index < len(future_lines):
            future_lines[index].set_visible(visible)
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

    plot_all_time_diffs(all_time_diffs, all_images, selected_folders, model_type)

if __name__ == "__main__":
    parent_folder = 'crops'
    target_color = np.array([0, 60, 0])  # Example target color

    # Let user choose the model type
    model_type = input("Choose model type (lstm/linear): ").lower()
    while model_type not in ['lstm', 'linear']:
        print("Invalid model type. Please choose 'lstm' or 'linear'")
        model_type = input("Choose model type (lstm/linear): ").lower()

    selected_folders = ask_user_for_folder_selection(parent_folder)
    plot_selected_folders(selected_folders, parent_folder, target_color, model_type)