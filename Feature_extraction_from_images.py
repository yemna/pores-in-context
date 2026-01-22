
main_folder = r'F:\Yemna\touching_vug'

import tensorflow as tf

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")    


import os
import cv2
import numpy as np
import tkinter as tk
import pywt
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import mahotas
import mahotas as mt
import mahotas
import tkinter as tk
import logging
from skimage.util import img_as_ubyte
from skimage import measure
from scipy import stats
from skimage.feature import canny
from skimage.feature import hog
from skimage.feature import local_binary_pattern, hog
from skimage.measure import label, regionprops
from skimage import measure
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.color import label2rgb
from scipy.ndimage import center_of_mass
from skimage import data, exposure
from scipy import stats
from tkinter import filedialog
from tkinter import simpledialog
from skimage.measure import moments_hu
from mahotas.features import zernike_moments
import numpy as np
import pandas as pd
import os
from skimage.measure import find_contours
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, ResNet50, InceptionResNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
from keras.applications.vgg19 import VGG19, preprocess_input as preprocess_vgg19
from tensorflow.keras.models import Model
# Transfer learning libraries
from keras.applications import VGG16
from keras.applications import VGG19
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications import VGG16, ResNet50, InceptionResNetV2, VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg19

print("Imports Loaded")

'''
def find_image_mask_pairs_corrected(folder_path):
    # Dictionary to store the pairs
    pairs = {}

    # Read the contents of the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            # Check if the file is a binary mask
            if '_bin_' in filename:
                # Extract the base name and label for the binary mask
                base_name, label = filename.replace('_bin_', '').r('_', 1) # original was _mask, changed it to _bin_ as that is the naming convention for the existing images with binary masks
                # Construct the original image name
                original_image_name = f"{base_name}_{label}"
                # Construct the full path for the image and the mask
                full_image_path = os.path.join(folder_path, original_image_name)
                full_mask_path = os.path.join(folder_path, filename)
                # Use the label as the key
                label = label.split(".")[0]
                label_key = int(label)  # Convert label to integer
                # Add the pair to the dictionary
                pairs[label_key] = {'image': full_image_path, 'mask': full_mask_path}

    return pairs
'''

# Configure TensorFlow to use GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured for use")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")


# Safe division function to avoid division by zero
def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else np.nan

def find_image_mask_pairs_corrected(folder_path):
    pairs = {}
    files = os.listdir(folder_path)
    
    # Find all RGB images and binary masks
    RGB_images = [f for f in files if ('cropped_label' in f) and f.endswith('.png')]
    binary_masks = [f for f in files if ('label_mask' in f) and f.endswith('.png')]
    
    print(f"\nFound {len(RGB_images)} RGB images and {len(binary_masks)} binary masks")
    
    for rgb_image in RGB_images:
        try:
            # Extract the parts from the RGB image name
            parts = rgb_image.split('_')
            number = parts[-1].replace('.png', '')
            
            # For both Modern and image patterns, preserve the exact prefix
            prefix = '_'.join(parts[:2])  # This will get "Modern_X" or "image_X"
            mask_filename = f"{prefix}_label_mask_{number}.png"
            
            # Check if corresponding mask exists
            if mask_filename in files:
                pairs[os.path.join(folder_path, rgb_image)] = os.path.join(folder_path, mask_filename)
                print(f"Found matching pair: {rgb_image} -> {mask_filename}")
            else:
                print(f"No matching mask found for RGB image: {rgb_image}")
                print(f"Looking for mask: {mask_filename}")
        except Exception as e:
            print(f"Error processing image {rgb_image}: {str(e)}")
    
    print(f"Successfully paired {len(pairs)} images")
    return pairs

def extract_statistics(image_path, mask_path):
    # Load the original image and the binary mask
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if original.shape != mask.shape:
      label = os.path.basename(image_path).split('.')[-2]  # Assumes label is the filename without the extension
      print(f"Dimension mismatch for label {label}: original {original.shape}, mask {mask.shape}")
      # Potentially resize one of the images here to match the other
      # For example, resizing mask to match original:
      # mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    else:

      # Apply the mask to get the pixel values of the region of interest
      region_pixels = original[mask == 255]
      
      if region_pixels.size > 0:
        if np.any(region_pixels < 0) or np.any(region_pixels > 255):
            print(f"Unexpected pixel value range detected in {image_path}")

      # Calculate the first-order statistics
      mean = np.mean(region_pixels) if region_pixels.size > 0 else None
      median = np.median(region_pixels) if region_pixels.size > 0 else None

      # Handling mode calculation using numpy's bincount
      if region_pixels.size > 0:
          pixel_counts = np.bincount(region_pixels)
          mode = np.argmax(pixel_counts)
      else:
          mode = None

      # Continue with other statistics
      minimum = np.min(region_pixels) if region_pixels.size > 0 else None
      maximum = np.max(region_pixels) if region_pixels.size > 0 else None
      range_ = maximum - minimum if region_pixels.size > 0 else None
      variance = np.var(region_pixels) if region_pixels.size > 0 else None
      std_dev = np.std(region_pixels) if region_pixels.size > 0 else None
      skewness = stats.skew(region_pixels, axis=None, bias=False) if region_pixels.size > 0 else None
      kurtosis = stats.kurtosis(region_pixels, axis=None, bias=False) if region_pixels.size > 0 else None
      entropy = stats.entropy(np.histogram(region_pixels, bins=256)[0]) if region_pixels.size > 0 else None
      
      # Ensure 'Maximum' and 'Minimum' are within the expected RGB range
      if minimum < 0 or maximum > 255:
          print(f"Pixel value range error in {image_path}: min {minimum}, max {maximum}")

      # Create a dictionary of the statistics
      statistics = {
          'Mean': mean,
          'Median': median,
          'Mode': mode,
          'Minimum': minimum,
          'Maximum': maximum,
          'Range': range_,
          'Variance': variance,
          'Standard Deviation': std_dev,
          'Skewness': skewness,
          'Kurtosis': kurtosis,
          'Entropy': entropy
      }

    return statistics

"""
def extract_statistics_to_csv(image_mask_pairs, output_csv_path):
    # List to store all the statistics
    all_statistics = []

    # Iterate over each image-mask pair in the dictionary
    for label, paths in image_mask_pairs.items():
        # Extract statistics for each pair
        image_path = paths['image']
        mask_path = paths['mask']
        stats = extract_statistics(image_path, mask_path)
        # Add the label to the statistics
        stats['Label'] = label
        # Append to the list
        all_statistics.append(stats)

    # Create a DataFrame from the list of statistics
    df = pd.DataFrame(all_statistics)


    # Reorder the DataFrame to have 'Label' as the first column
    column_order = ['Label'] + [col for col in df if col != 'Label']
    #print(df.columns)

    df = df[column_order]
    #print(df.columns)

    # Save the DataFrame as a CSV file to the specified path
    df.to_csv(output_csv_path, index=False)
"""

def extract_statistics_to_csv(image_mask_pairs, output_csv_path):
    # Assuming 'extract_statistics' is a function that takes paths to an image and its mask and returns calculated statistics as a dictionary
    all_statistics = []

    for image_path, mask_path in image_mask_pairs.items():
        # Extract statistics for each image-mask pair
        stats = extract_statistics(image_path, mask_path)

        # Assuming 'stats' is a dictionary, you might need to add additional info, such as the image label or path, if necessary
        #stats['Image_Path'] = image_path  # For example, adding the image path to the statistics
        
        # Extracting label from the image path
        label = os.path.splitext(os.path.basename(image_path))[0].replace('_RGB', '')
        stats['Label'] = label  # Ensuring consistent 'label' column naming
        all_statistics.append(stats)

    # Convert the list of statistics dictionaries to a DataFrame
    df = pd.DataFrame(all_statistics)

    # Save the DataFrame to CSV
    df.to_csv(output_csv_path, index=False)
    #append_df_to_csv(df, output_csv_path)  # Use the new function to append data
    
def extract_lbp_features(image, P=24, R=8):
    # Check if the image is 2-dimensional, convert to RGB if not
    if len(image.shape) > 2:
        # Convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Local Binary Pattern representation
    lbp = local_binary_pattern(image, P, R, method="uniform")
    n_bins = int(lbp.max() + 1)
    # Calculate the LBP histogram
    lbp_hist = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)[0]

    # Create a dictionary with separate entries for each histogram bin
    lbp_features = {f'LBP_P{P}_R{R}_bin{i}': bin_val for i, bin_val in enumerate(lbp_hist)}

    return lbp_features

def get_haralick_feature_names():
    return [
        'AngularSecondMoment',
        'Contrast',
        'Correlation',
        'SumOfSquares',
        'InverseDifferenceMoment',
        'SumAverage',
        'SumVariance',
        'SumEntropy',
        'Entropy',
        'DifferenceVariance',
        'DifferenceEntropy',
        'InformationMeasureOfCorrelation1',
        'InformationMeasureOfCorrelation2',
        'MaximalCorrelationCoefficient'
    ]

def extract_haralick_features(image):
    # Get the names of Haralick features from the predefined function
    haralick_names = get_haralick_feature_names()
    
    if np.count_nonzero(image) == 0:  # Check if the image is almost empty
        # Use a fixed number of features based on typical Haralick feature count
        return {f'{haralick_names[i]}_D{distance}': np.nan for distance in range(1, 21) for i in range(len(haralick_names))}


    # Attempt to compute Haralick features at distance 1 to determine feature count
    try:
        initial_features = mahotas.features.haralick(image, distance=1).mean(axis=0)
        num_features = len(initial_features)
    except ValueError:
        # If even the initial computation fails, provide NaNs for all expected features
        return {f'{haralick_names[i]}_D{distance}': np.nan for distance in range(1, 21) for i in range(len(haralick_names))}


    haralick_features_dict = {}
    for distance in range(1, 21):
        try:
            # Compute Haralick features for the current distance
            features = mahotas.features.haralick(image, distance=distance).mean(axis=0)
            for i in range(num_features):
                haralick_features_dict[f'{haralick_names[i]}_D{distance}'] = features[i]
        except ValueError:
            # If features cannot be computed for this distance, fill in NaNs
            for i in range(num_features):
                haralick_features_dict[f'{haralick_names[i]}_D{distance}'] = np.nan

    return haralick_features_dict



def extract_zernike_moments(image):
    if not image.any():
        return {"Zernike_None": np.nan}  # Handling empty or invalid images

    try:
        # Ensure the image is 2D
        if image.ndim == 3 and image.shape[-1] == 1:
            # If image has a singleton third dimension, remove it
            image = image[:, :, 0]
        elif image.ndim == 3:
            # Convert the image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ensure the image is binary
        binary_image = image > 0

        # Calculate the centroid of the object
        centroid = center_of_mass(binary_image)
        if len(centroid) != 2:
            raise ValueError("Image is not 2D after processing.")

        y_centroid, x_centroid = centroid

        # Find coordinates of all points in the object
        y, x = np.where(binary_image)

        # Corrected distance calculation
        distances = np.sqrt((x - x_centroid)**2 + (y - y_centroid)**2)

        # Find the maximum distance
        max_distance = distances.max()

        # Calculate the Zernike moments for the calculated radius
        zernike_features_array = mahotas.features.zernike_moments(binary_image, int(max_distance))

        # Create a dictionary with separate entries for each Zernike moment
        zernike_features_dict = {f'Zernike_{i}': moment for i, moment in enumerate(zernike_features_array, 1)}

        return zernike_features_dict
    except Exception as e:
        # Handle exceptions and errors
        return {"Zernike_Error": str(e)}

def get_fft_features(data):
    # Avoid log2(0) by setting a minimum value
    data_safe = np.where(data > 0, data, 1e-10)
    return {
        'Mean': np.mean(data),
        'Min': np.min(data),
        'Max': np.max(data),
        'Median': np.median(data),
        'Std_dev': np.std(data),
        'Variance': np.var(data),
        'Range': np.ptp(data),
        'Energy': np.sum(data ** 2),
        'Entropy': -np.sum(data_safe * np.log2(data_safe))
    }

def compute_transforms(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to RGB
    image = np.float32(image) / 255.0  # Normalize the image

    # Compute DFT
    dft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Compute magnitude spectrum and apply log transformation for visualization
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

    # Extract FFT features and flatten them into the output dictionary
    fft_features = get_fft_features(magnitude_spectrum)

    # Prefix each key with 'FFT_' to indicate these features come from the FFT
    fft_features = {f'FFT_{key}': value for key, value in fft_features.items()}

    return fft_features

def get_dwt_features_for_wavelets(image):
    wavelets = [
        ('db', list(range(1, 21, 5))),  # Daubechies
        ('sym', list(range(2, 21, 5))),  # Symlets
        ('coif', list(range(1, 6)))  # Coiflets
    ]

    dwt_features = {}

    # Process multi-parameter wavelets
    for base, nums in wavelets:
        for num in nums:
            wavelet_name = f"{base}{num}"
            dwt_features.update(get_dwt_features(image, wavelet_name))

    return dwt_features

def get_dwt_features(image, wavelet_name):
    # Apply Discrete Wavelet Transform with the specified wavelet
    wavelet_coeffs = pywt.dwt2(image, wavelet_name)
    cA, (cH, cV, cD) = wavelet_coeffs

    # Flatten the feature dictionaries so each feature becomes a separate column
    features = {}
    for component, data in zip(['Approximation', 'Horizontal_Detail', 'Vertical_Detail', 'Diagonal_Detail'], [cA, cH, cV, cD]):
        stats = calculate_stats(data)
        for stat_name, stat_value in stats.items():
            features[f'{wavelet_name}_{component}_{stat_name}'] = stat_value

    return features

def calculate_stats(data):
    data_safe = np.where(data > 0, data, np.nextafter(0, 1))  # Use the smallest positive float to replace zeros
    return {
        'Mean': np.mean(data),
        'Min': np.min(data),
        'Max': np.max(data),
        'Median': np.median(data),
        'Std dev': np.std(data),
        'Variance': np.var(data),
        'Range': np.ptp(data),
        'Energy': np.sum(data ** 2),
        'Entropy': -np.sum(np.where(data_safe > 0, data_safe * np.log2(data_safe), 0))
    }

def extract_simple_shape_features(image, mask):
    labeled_image = measure.label(mask)
    regions = measure.regionprops(labeled_image, intensity_image=image)

    # Assuming we are interested in the largest region
    if not regions:
        return {}

    largest_region = max(regions, key=lambda r: r.area)
    feret_diameter_max = largest_region['feret_diameter_max']
    
    features = {
        'Eccentricity': largest_region.eccentricity,
        'Centroid': largest_region.centroid,
        'Solidity': largest_region.solidity,
        'Extent': largest_region.extent,
        'Euler Number': largest_region.euler_number,
        'FormFactor': safe_divide(4.0 * np.pi * largest_region.area, largest_region.perimeter ** 2),
        'Aspect Ratio': safe_divide(largest_region.major_axis_length, largest_region.minor_axis_length),
        #'Radius Ratio': safe_divide(largest_region.equivalent_diameter, largest_region.solidity),
        #'Elongation': safe_divide(largest_region.major_axis_length, largest_region.minor_axis_length),
        'Roundness': safe_divide(4 * largest_region.area, np.pi * feret_diameter_max ** 2),
        'Area': largest_region.area,
        'Perimeter': largest_region.perimeter,
        'Compactness': safe_divide(np.sqrt(4 * largest_region.area / np.pi), feret_diameter_max),
        'Axis Major Length': largest_region.major_axis_length,
        'Axis Minor Length': largest_region.minor_axis_length,
        'Max Feret Diameter': feret_diameter_max,
        'Orientation': largest_region.orientation
    }
    return features

# Fourier shape descriptors
def extract_contour(mask):
    """ Extract the largest contour from the binary mask. """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def fourier_descriptors(contour, num_descriptors=20):
    """Compute Fourier Descriptors for a contour and return as a dictionary with magnitudes."""
    contour = contour.squeeze()
    complex_contour = np.empty(contour.shape[:-1], dtype=complex)
    complex_contour.real = contour[:, 0]
    complex_contour.imag = contour[:, 1]
    fourier_result = np.fft.fft(complex_contour)
    descriptors = np.fft.fftshift(fourier_result)

    # Keep only a certain number of descriptors
    center_index = len(descriptors) // 2
    descriptors = descriptors[center_index - num_descriptors // 2:center_index + num_descriptors // 2]

    # Compute the magnitudes of the descriptors
    magnitudes = np.abs(descriptors)

    # Create a dictionary with separate entries for each descriptor's magnitude
    descriptor_dict = {f'Fourier_Descriptor_{i}': magnitude for i, magnitude in enumerate(magnitudes, 1)}

    return descriptor_dict
def extract_contour(mask):
    """Find the largest contour in the mask and return it."""
    contours = find_contours(mask, level=0.5)
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=lambda x: x.shape[0])
        return largest_contour
    else:
        return None

# Function to load and process the image
def load_process_image(image_path, mask_path):
    # Load the image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)  # Assuming mask is RGB

    # Resize the image and mask to 112x112
    image_resized = cv2.resize(image, (112, 112))
    mask_resized = cv2.resize(mask, (112, 112))

    # Apply the mask if necessary (optional based on your specific requirement)
    masked_image = cv2.bitwise_and(image_resized, image_resized, mask=mask_resized)

    # Convert to the format expected by VGG16 (e.g., BGR to RGB, float32, etc.)
    processed_image = preprocess_input(masked_image.astype('float32'))

    # Ensure the image has the correct shape (112, 112, 3)
    assert processed_image.shape == (112, 112, 3), f"Processed image has incorrect shape {processed_image.shape}"

    # Add an extra dimension to match the input shape expected by VGG16, i.e., (None, 112, 112, 3)
    processed_image = np.expand_dims(processed_image, axis=0)

    return processed_image

def extract_features_for_pairs(image_mask_pairs, output):
    all_features = []
    total_labels = len(image_mask_pairs)
    
    # Import additional models
    from tensorflow.keras.applications import DenseNet121, EfficientNetB4
    from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
    from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
    from tensorflow.keras.layers import GlobalAveragePooling2D
    
    # Load models once, outside the loop
    vgg16_base = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    resnet50_base = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    inceptionresnetv2_base = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(299, 299, 3))
    vgg19_base = VGG19(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    densenet121_base = DenseNet121(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    efficientnet_base = EfficientNetB4(include_top=False, weights="imagenet", input_shape=(380, 380, 3))

    # Add GlobalAveragePooling2D layer to each model
    gap_layer = GlobalAveragePooling2D()
    
    # Create models with global average pooling
    models = {
        'VGG16': tf.keras.Sequential([vgg16_base, gap_layer]),  # Will output 512 features
        'VGG19': tf.keras.Sequential([vgg19_base, gap_layer]),  # Will output 512 features
        'ResNet50': tf.keras.Sequential([resnet50_base, gap_layer]),  # Will output 2048 features
        'InceptionResNetV2': tf.keras.Sequential([inceptionresnetv2_base, gap_layer]),  # Will output 1536 features
        'DenseNet121': tf.keras.Sequential([densenet121_base, gap_layer]),  # Will output 1024 features
        'EfficientNetB4': tf.keras.Sequential([efficientnet_base, gap_layer])  # Will output 1792 features
    }

    # Define input sizes and preprocessing functions for each model
    model_configs = {
        'VGG16': {'size': (224, 224), 'preprocess': preprocess_vgg16},
        'ResNet50': {'size': (224, 224), 'preprocess': preprocess_resnet50},
        'InceptionResNetV2': {'size': (299, 299), 'preprocess': preprocess_inceptionresnetv2},
        'VGG19': {'size': (224, 224), 'preprocess': preprocess_vgg19},
        'DenseNet121': {'size': (224, 224), 'preprocess': preprocess_densenet},
        'EfficientNetB4': {'size': (380, 380), 'preprocess': preprocess_efficientnet}
    }

    # Optimize prediction function
    @tf.function
    def predict_features(model, image):
        return model(image, training=False)

    for current_label_number, (image_path, mask_path) in enumerate(image_mask_pairs.items(), 1):
        try:
            label = os.path.splitext(os.path.basename(image_path))[0].replace('_RGB', '')
            print(f"Processing label: {label}, {current_label_number} of {total_labels}")

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            mask = cv2.imread(mask_path, 0)
            if mask is None:
                raise ValueError(f"Failed to load mask: {mask_path}")

            masked_image = cv2.bitwise_and(image, image, mask=mask)
            
            features = {'Label': label}

            # Extract handcrafted features
            print("Extracting simple shape features...")
            features.update(extract_simple_shape_features(image, mask))

            print("Extracting LBP features...")
            features.update(extract_lbp_features(masked_image))
            features.update(extract_lbp_features(masked_image, P=16, R=2))
            features.update(extract_lbp_features(masked_image, P=8, R=1))

            print("Extracting Haralick features...")
            features.update(extract_haralick_features(masked_image))

            print("Extracting Zernike moments...")
            features.update(extract_zernike_moments(masked_image))

            print("Extracting DWT features...")
            features.update(get_dwt_features_for_wavelets(image))

            print("Extracting FFT features...")
            features.update(compute_transforms(masked_image))

            print("Extracting Fourier Shape Descriptors...")
            contour = extract_contour(mask)
            if contour is not None and len(contour) > 0:
                fourier_features = fourier_descriptors(contour, num_descriptors=20)
                features.update(fourier_features)
            else:
                print("No contours found in the mask.")

            # Extract deep learning features using GPU
            print("Extracting DL features...")
            for model_name, model in models.items():
                config = model_configs[model_name]
                size = config['size']
                preprocess = config['preprocess']
                
                # Resize and preprocess image
                processed_image = cv2.resize(masked_image, size)
                processed_image = preprocess(processed_image.astype('float32'))
                processed_image = np.expand_dims(processed_image, axis=0)
                
                # Extract features with global average pooling
                dl_features = predict_features(model, processed_image).numpy().flatten()
                
                # Add features to dictionary with model name prefix
                features.update({f"{model_name}_{i}": feature for i, feature in enumerate(dl_features)})

            all_features.append(features)

        except Exception as e:
            error_message = f"Error processing image pair: {image_path}, {mask_path}. Error: {str(e)}"
            print(error_message)
            logging.error(error_message)
            continue

    # Create a DataFrame from all the extracted features
    df = pd.DataFrame(all_features)

    # Save the DataFrame to CSV
    df.to_csv(output, index=False)

    print("Feature extraction complete.")

def get_data_subfolders(data_folder=main_folder):
    print("Scanning for subfolders in the 'Data' directory...")
    subfolders = []
    for name in os.listdir(data_folder):
        full_path = os.path.join(data_folder, name)
        if os.path.isdir(full_path):
            non_uniform_path = os.path.join(full_path, '_Non-uniform')
            if os.path.exists(non_uniform_path):
                subfolders.append((name, non_uniform_path))  # Store tuple of (parent folder name, _Non-uniform subfolder path)
    return subfolders

def process_subfolder(subfolder_info, results_root='Extracted_features'):
    parent_folder_name, non_uniform_path = subfolder_info
    print(f"\nProcessing data from: {parent_folder_name}")

    results_subfolder_path = ensure_results_subfolder(parent_folder_name, results_root)
    image_mask_pairs = find_image_mask_pairs_corrected(non_uniform_path)

    # Define the output CSV file paths
    fo_output_csv_path = os.path.join(results_subfolder_path, f"{parent_folder_name}_First_Order_Features.csv")
    so_output_csv_path = os.path.join(results_subfolder_path, f"{parent_folder_name}_Second_Order_Features.csv")
    merged_output_csv_path = os.path.join(results_subfolder_path, f"{parent_folder_name}_Merged_Features.csv")

    print(f"Extracting features to: {fo_output_csv_path} and {so_output_csv_path}")
    
    # First Order Features
    extract_statistics_to_csv(image_mask_pairs, fo_output_csv_path)
    
    # Second Order Features
    extract_features_for_pairs(image_mask_pairs, so_output_csv_path)
    
    # Merge first and second order features into one CSV
    merge_csv_files(fo_output_csv_path, so_output_csv_path, merged_output_csv_path)

    print(f"Features extracted and saved to {fo_output_csv_path} and {so_output_csv_path}")
    print(f"Merged features saved to {merged_output_csv_path}")



def ensure_results_subfolder(data_subfolder, results_root='../Extracted_features'):
    results_subfolder_name = os.path.basename(data_subfolder)
    results_subfolder_path = os.path.join(results_root, results_subfolder_name)
    if not os.path.exists(results_subfolder_path):
        os.makedirs(results_subfolder_path)
        print(f"Created results subfolder: {results_subfolder_path}")
    else:
        print(f"Results subfolder already exists: {results_subfolder_path}")
    return results_subfolder_path

def merge_csv_files(fo_csv_path, so_csv_path, output_csv_path):
    # Load the CSV files into DataFrames
    fo_df = pd.read_csv(fo_csv_path)
    so_df = pd.read_csv(so_csv_path)
    
    # Print unique labels to ensure they match
    print("Unique labels in FO DataFrame:", fo_df['Label'].unique())
    print("Unique labels in SO DataFrame:", so_df['Label'].unique())
    
    # Rename FO columns to end with '_FO', except for 'Label'
    fo_df.rename(columns=lambda x: x + '_FO' if x != 'Label' else x, inplace=True)

    # Merge the DataFrames on the 'Label' column
    merged_df = pd.merge(fo_df, so_df, on='Label')

    # Save the merged DataFrame to a new CSV
    merged_df.to_csv(output_csv_path, index=False)

# Main script adjustments
#data_subfolders = get_data_subfolders()
#print(f"Subfolders: {data_subfolders}")
#for subfolder_info in data_subfolders:
#    process_subfolder(subfolder_info)

#print("Feature extraction completed!")


### This part of code has been added
# Create the output directory if it doesn't exist
results_root = r'F:\Yemna\touching_vug\Extracted_features'
if not os.path.exists(results_root):
    os.makedirs(results_root)

# Get image-mask pairs directly from the main folder
image_mask_pairs = find_image_mask_pairs_corrected(main_folder)

if image_mask_pairs:
    # Define output paths
    fo_output_csv_path = os.path.join(results_root, "First_Order_Features.csv")
    so_output_csv_path = os.path.join(results_root, "Second_Order_Features.csv")
    merged_output_csv_path = os.path.join(results_root, "Merged_Features.csv")

    print(f"Found {len(image_mask_pairs)} image-mask pairs")
    print(f"Extracting features to: {fo_output_csv_path} and {so_output_csv_path}")
    
    # Extract features
    extract_statistics_to_csv(image_mask_pairs, fo_output_csv_path)
    extract_features_for_pairs(image_mask_pairs, so_output_csv_path)
    
    # Merge the features
    merge_csv_files(fo_output_csv_path, so_output_csv_path, merged_output_csv_path)

    print(f"Features extracted and saved to {fo_output_csv_path} and {so_output_csv_path}")
    print(f"Merged features saved to {merged_output_csv_path}")
else:
    print("No image-mask pairs found in the directory")

print("Feature extraction completed!")