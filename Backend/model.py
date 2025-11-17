import sys
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

# ---------------------- Load Datasets ----------------------
def load_datasets():
    """Loads and validates the necessary datasets."""
    try:
        # Assuming datasets are in the 'datasets' folder relative to where the script is run
        palette_data = pd.read_excel("datasets/color_palate_ds.xlsx")
        suggestions = pd.read_excel("datasets/suggestions.xlsx")
        
        palette_data.columns = palette_data.columns.str.strip()
        suggestions.columns = suggestions.columns.str.strip()
        
        required_cols_palette = ['Season', 'Red', 'Green', 'Blue', 'Hex_Code']
        # Ensure 'Gender' column is present in suggestions for frontend filtering
        required_cols_suggestions = ['Season', 'Gender', 'Style', 'Product Link', 'Image Path'] 
        
        if not all(col in palette_data.columns for col in required_cols_palette):
            raise ValueError(f"Missing required columns in color_palate_ds.xlsx. Required: {required_cols_palette}")
        if not all(col in suggestions.columns for col in required_cols_suggestions):
            raise ValueError(f"Missing required columns in suggestions.xlsx. Required: {required_cols_suggestions}")
            
        return palette_data, suggestions
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure 'datasets/color_palate_ds.xlsx' and 'datasets/suggestions.xlsx' exist.")
        return None, None
    except ValueError as e:
        print(f"Error: {e}")
        return None, None

palette_data, suggestions = load_datasets()

# ---------------------- Extract Skin Color ----------------------
def extract_skin_color(image_path):
    """
    Extracts the dominant skin color (RGB) from an image after applying CLAHE.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image file.")

    # Convert image to LAB color space
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)

    # Apply CLAHE to the L channel for enhanced contrast
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8)) 
    clahe_l_channel = clahe.apply(l_channel)
    clahe_lab_img = cv2.merge([clahe_l_channel, a_channel, b_channel])
    clahe_bgr_img = cv2.cvtColor(clahe_lab_img, cv2.COLOR_LAB2BGR)

    # Simplified ROI (Region of Interest) for the face/skin area
    height, width, _ = clahe_bgr_img.shape
    # Taking a central 100x100 pixel area
    roi = clahe_bgr_img[height//2 - 50:height//2 + 50, width//2 - 50:width//2 + 50]
    
    # Reshape and cluster pixels to find the dominant color
    pixels = roi.reshape(-1, 3)
    # n_clusters=1 finds the average color of the ROI
    # n_init=10 is used for robust K-Means initialization
    kmeans = KMeans(n_clusters=1, random_state=42, n_init=10).fit(pixels)
    # The result is in BGR, convert to RGB for consistency with pandas data
    dominant_color_bgr = kmeans.cluster_centers_[0].astype(int)
    dominant_color_rgb = dominant_color_bgr[::-1]
    
    return dominant_color_rgb.tolist()

# ---------------------- Predict Season ----------------------
def predict_undertone(skin_rgb):
    """
    Predicts the season based on the skin's RGB color using Euclidean distance.
    """
    if palette_data is None:
        raise RuntimeError("Palette data is not loaded.")

    min_dist = float('inf')
    best_match = None
    palette_colors_df = palette_data[['Season', 'Red', 'Green', 'Blue']] 
    
    for _, row in palette_colors_df.iterrows():
        # Using the Red, Green, Blue columns from the dataset
        palette_rgb = [row['Red'], row['Green'], row['Blue']]
        dist = euclidean(skin_rgb, palette_rgb)
        if dist < min_dist:
            min_dist = dist
            best_match = row['Season']
    
    return best_match

# ---------------------- Get Season Palette Colors ----------------------
def get_season_palette_colors(season):
    """
    Retrieves all hex codes and RGB values for a given season from the palette data, 
    formatted as a list of dictionaries for frontend use.
    """
    if palette_data is None:
        raise RuntimeError("Palette data is not loaded.")
    
    season_colors = palette_data[palette_data['Season'].str.lower() == season.lower()]
    
    color_list = []
    for _, row in season_colors.iterrows():
        color_list.append({
            'hex_code': row['Hex_Code'],
            # CRITICAL FIX: Add a 'name' field for the frontend hover text
            'name': row.get('Color_Name', row['Hex_Code']), # Assuming 'Color_Name' exists, else use Hex
            'rgb': [int(row['Red']), int(row['Green']), int(row['Blue'])]
        })
    return color_list

# ---------------------- Get Suggestions ----------------------
def get_suggestions(season):
    """
    Retrieves a list of suggestions for a given season.
    """
    if suggestions is None:
        raise RuntimeError("Suggestions data is not loaded.")
    
    season_suggestions = suggestions[suggestions['Season'].str.lower() == season.lower()]
    
    # The 'Image Path' column name needs to match your dataset exactly
    image_path_col = 'Image Path' 
    if image_path_col not in season_suggestions.columns:
         # Fallback if 'Image Path' is not the exact column name, try to find one containing 'image'
        image_path_col = next((col for col in season_suggestions.columns if 'image' in col.lower()), None)
        if not image_path_col:
            raise ValueError("No column containing 'image' found in the suggestions dataset.")
    
    suggestions_list = []
    for _, row in season_suggestions.iterrows():
        suggestions_list.append({
            'gender': row.get('Gender'), # Assuming 'Gender' column exists in your suggestions.xlsx
            'style': row.get('Style'),
            'image_path': row.get(image_path_col),
            'product_link': row.get('Product Link')
        })
    return suggestions_list
    
# ---------------------- Get ALL Palette Colors (MODIFIED) ----------------------
def get_all_palette_colors():
    """Retrieves all color data (Season, Hex, RGB) from the dataset."""
    if palette_data is None:
        raise RuntimeError("Palette data is not loaded.")
    
    all_colors = []
    for _, row in palette_data.iterrows():
        all_colors.append({
            'season': row['Season'],
            'hex_code': row['Hex_Code'],
            # Adding a name field for better frontend compatibility
            'name': row['Hex_Code'], 
            'rgb': [int(row['Red']), int(row['Green']), int(row['Blue'])]
        })
    return all_colors


# ---------------------- Combined Prediction Function for API (CORRECTED) ----------------------
def analyze_image_and_get_data(image_path):
    """
    Performs the full analysis pipeline and returns ALL data required by the frontend modal.
    """
    try:
        # 1. Extract skin color
        skin_rgb = extract_skin_color(image_path)
        
        # 2. Predict season
        season = predict_undertone(skin_rgb)
        if not season:
            return None, "Could not determine a season match."
        
        # 3. Get predicted season palette and suggestions
        palette = get_season_palette_colors(season)
        style_suggestions = get_suggestions(season)
        
        # 4. GET ALL COLORS (CRITICAL FOR THE TRY-ON GRID)
        all_colors = get_all_palette_colors()

        # Return all data fields expected by the frontend
        return {
            'dominant_skin_color_rgb': skin_rgb,
            'predicted_season': season,
            'palette_colors': palette,
            'suggestions': style_suggestions,
            'all_palette_colors': all_colors  # <--- NEW REQUIRED FIELD
        }, None
        
    except Exception as e:
        return None, str(e)