import os
import cv2
import numpy as np
from PIL import Image


def load_images_from_folder(folder_path):
    images = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(root, filename)
                try:
                    image = Image.open(image_path)
                    images.append(image)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    return images


def preprocess_image(image):
    # Convert PIL Image to NumPy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale if the image is RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply adaptive threshold to separate background and foreground (more robust for various lighting)
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    return binary_image


def segment_texts(image):
    # Morphological operations to separate printed text from handwritten text
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find connected components (regions of text)
    num_labels, labels_im = cv2.connectedComponents(morphed)

    # Extract each component and return masks for handwritten and printed text
    printed_text_mask = np.zeros(image.shape, dtype=np.uint8)
    handwritten_text_mask = np.zeros(image.shape, dtype=np.uint8)

    # Iterate over all components and classify based on size or shape (handwritten is denser)
    for label in range(1, num_labels):
        component_mask = np.uint8(labels_im == label) * 255
        area = cv2.countNonZero(component_mask)

        if area > 5000:  # Assuming printed text is larger and less dense
            printed_text_mask = cv2.bitwise_or(printed_text_mask, component_mask)
        else:
            handwritten_text_mask = cv2.bitwise_or(handwritten_text_mask, component_mask)

    return printed_text_mask, handwritten_text_mask


def apply_mask_and_save(image_np, mask, output_path):
    if len(image_np.shape) == 2:  # If grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    result = cv2.bitwise_and(image_np, image_np, mask=mask)

    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(output_path, result_gray)


def extract_and_save_texts(image_folder, output_dir):
    images = load_images_from_folder(image_folder)

    for idx, image in enumerate(images):
        preprocessed_image = preprocess_image(image)

        printed_text_mask, handwritten_text_mask = segment_texts(preprocessed_image)

        image_np = np.array(image)

        if printed_text_mask is not None:
            apply_mask_and_save(image_np, printed_text_mask, f'{output_dir}/printed_text_{idx}.png')

        if handwritten_text_mask is not None:
            apply_mask_and_save(image_np, handwritten_text_mask, f'{output_dir}/handwritten_text_{idx}.png')


# Example usage:
extract_and_save_texts('data', 'extracted')
