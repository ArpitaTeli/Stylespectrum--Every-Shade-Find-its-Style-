import os
import sys 
# Import necessary modules
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import model # Explicitly import the model module

# Ensure model imports are correct, assuming they are available
try:
    # Assuming model.py is in the same directory
    from model import analyze_image_and_get_data, get_all_palette_colors, get_suggestions
except ImportError:
    print("FATAL: Could not import functions from model.py.", file=sys.stderr)
    analyze_image_and_get_data = None
    get_all_palette_colors = None
    get_suggestions = None


app = Flask(__name__)

# 1. Define the temporary upload folder for camera capture blobs
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 2. Define the static folder for suggestion images
IMAGE_DIR = os.path.join(app.root_path, 'static', 'images') 
os.makedirs(IMAGE_DIR, exist_ok=True) 

# Check if the required datasets were loaded in model.py
if analyze_image_and_get_data is None or get_all_palette_colors is None:
    print("WARNING: Model analysis functions are missing. The /api/analyze_image endpoint will likely fail.", file=sys.stderr)
    pass 

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

# --- FIX 2: REMOVED CONFLICTING '/image/<filename>' ROUTE ---
# Flask's default static serving handles /static/images/ automatically,
# or we let the main API endpoint handle the path transformation.


@app.route('/api/analyze_image', methods=['POST']) 
def analyze_image():
    """
    Handles the POST request containing the camera-captured image blob 
    and returns the complete analysis results (season, palette, and suggestions).
    """
    if analyze_image_and_get_data is None:
        return jsonify({'error': 'Analysis service is unavailable (Model not loaded).'}), 503
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided from camera capture. Frontend error?'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file or empty file name.'}), 400

    file_path = None
    
    try:
        # --- File Saving Logic (REQUIRED for cv2.imread) ---
        filename = secure_filename(file.filename)
        unique_filename = f"{os.urandom(8).hex()}_{filename}" 
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
             return jsonify({'error': 'Failed to save captured image blob to server disk.'}), 500

        # --- Call the analysis function from model.py ---
        results, error = analyze_image_and_get_data(file_path)
        
        if error:
            return jsonify({'error': f'Analysis failed in model.py: {error}'}), 500
        
        # --- Fetch supplementary data ---
        if get_all_palette_colors:
            all_colors = get_all_palette_colors() 
        else:
            all_colors = []
            
        if results:
            # CRITICAL FIX 3: Manually transform image paths for suggestions data
            if 'suggestions' in results and results['suggestions']:
                for item in results['suggestions']:
                    if item.get('image_path'):
                        # This transforms 'jetblack.png' into '/static/images/jetblack.png'
                        item['image_path'] = f'/static/images/{item["image_path"]}'
            
            # Add all colors to the results dictionary
            results['all_palette_colors'] = all_colors 
            
        return jsonify(results), 200

    except Exception as e:
        print(f"Server error during analysis: {e}", file=sys.stderr)
        return jsonify({'error': f'An unexpected server error occurred on the backend: {e}'}), 500
    
    finally:
        # --- Cleanup Logic: Delete the temporary file ---
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up file {file_path}: {e}", file=sys.stderr)

# --- FIX 4: REMOVED SEPARATE get_suggestions ROUTE ---
# The logic for suggestions is now consolidated into analyze_image().


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
