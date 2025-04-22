import cv2
import numpy as np

def color_distance(c1, c2):
    return np.sqrt(np.sum((c1 - c2) ** 2))

def filter_closest_colors(image_path, target_color, threshold=50):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    target_color = np.array(target_color)
    mask = np.apply_along_axis(lambda px: color_distance(px, target_color) < threshold, 2, image)
    
    filtered_image = np.zeros_like(image)
    filtered_image[mask] = image[mask]
    
    return image, filtered_image

def main():
    image_path = r'crops\living_0.5%\frame_capture_D(30_1_2025)_T(8_23).jpg_crop_living_0.5%.png'  # Change to your image file
    target_color = (0, 30, 0)  # Example: Red color
    threshold = 50  # Adjust for more or less strict color matching
    
    original_image, result = filter_closest_colors(image_path, target_color, threshold)
    
    avg_color = np.mean(original_image, axis=(0, 1)).astype(int)
    avg_color_image = np.full_like(original_image, avg_color)
    
    cv2.imshow('Original Image', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Filtered Image', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.imshow('Average Color Image', cv2.cvtColor(avg_color_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
