import os
import logging
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import cv2
import numpy as np
import tempfile
import threading
from werkzeug.utils import secure_filename

from utils.detection import detect_violations, save_detection_result
from utils.email_sender import send_violation_email

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key-for-development")

# Configure upload folder
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if input type is selected
        input_type = request.form.get('inputType')
        if not input_type:
            flash('Please select input type (Image or Video)', 'error')
            return redirect(url_for('index'))
        
        # Check if file part exists
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        # Check if user submitted an empty form
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for('index'))
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            flash(f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}', 'error')
            return redirect(url_for('index'))
        
        # Generate a unique filename to avoid conflicts
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(filepath)
        logger.debug(f"File saved as {filepath}")
        
        # Process the file for violations
        email_address = "jabsmeher@gmail.com"  # Hardcoded as per requirements
        
        # Process the file based on type
        is_video = file_extension in ['mp4', 'avi', 'mov']
        
        # Detect violations
        violations, output_path = detect_violations(
            filepath, 
            is_video=is_video,
            original_filename=original_filename
        )
        
        # If violations detected, send email in background to not block the response
        if violations:
            threading.Thread(
                target=send_violation_email,
                args=(email_address, violations, original_filename)
            ).start()
        
        # Store result in session for display
        session['detection_result'] = {
            'filename': original_filename,
            'violations': violations,
            'email_sent': bool(violations),
            'email_address': email_address if violations else None,
            'output_path': output_path
        }
        
        return redirect(url_for('show_result'))
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/result')
def show_result():
    # Get result from session
    result = session.get('detection_result')
    if not result:
        flash('No detection results found. Please upload a file first.', 'error')
        return redirect(url_for('index'))
    
    return render_template('result.html', result=result)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"})

# Clean up temporary files periodically
@app.teardown_appcontext
def cleanup_temp_files(error):
    # This is a simple implementation - in a production system, 
    # you'd want a more sophisticated approach
    import os
    import time
    
    current_time = time.time()
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Delete files older than 1 hour
        if os.path.isfile(filepath) and os.path.getmtime(filepath) < current_time - 3600:
            try:
                os.remove(filepath)
                logger.debug(f"Cleaned up temporary file: {filepath}")
            except Exception as e:
                logger.error(f"Error cleaning up file {filepath}: {str(e)}")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
