import os
import cv2
import numpy as np
import logging
import tempfile
import urllib.request
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Base directory for models
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils', 'models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# YOLO model paths
YOLO_WEIGHTS = os.path.join(MODEL_DIR, 'yolov4-tiny.weights')
YOLO_CONFIG = os.path.join(MODEL_DIR, 'yolov4-tiny.cfg')
COCO_NAMES = os.path.join(MODEL_DIR, 'coco.names')

# Output directory for processed files
OUTPUT_DIR = 'static/output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Check if models exist, if not, download them
def check_and_download_models():
    """Check if YOLO models exist, download if not."""
    if not os.path.exists(YOLO_WEIGHTS):
        logger.info("Downloading YOLOv4-tiny weights...")
        urllib.request.urlretrieve(
            'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights',
            YOLO_WEIGHTS
        )
    
    if not os.path.exists(YOLO_CONFIG):
        logger.info("Downloading YOLOv4-tiny config...")
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg',
            YOLO_CONFIG
        )
    
    if not os.path.exists(COCO_NAMES):
        logger.info("Downloading COCO names...")
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names',
            COCO_NAMES
        )

# Load YOLO network
def load_yolo():
    """Load YOLO network."""
    check_and_download_models()
    
    # Load YOLO network
    net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG, YOLO_WEIGHTS)
    
    # Use CPU if CUDA is not available
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    
    # Load class names
    with open(COCO_NAMES, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, output_layers, classes

def detect_objects(frame, net, output_layers, classes):
    """
    Detect objects in a frame using YOLO.
    
    Args:
        frame: Input image frame
        net: Pre-loaded YOLO network
        output_layers: Output layer names
        classes: Class names
        
    Returns:
        detected_objects: List of detected objects with coordinates
    """
    height, width = frame.shape[:2]
    
    # Detect objects
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    # Process detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # Confidence threshold
                # Scale the bounding box coordinates to the image size
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype("int")
                
                # Calculate top-left corner of bounding box
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))
                
                # Add detection
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            detected_objects.append({
                'class': classes[class_ids[i]],
                'confidence': confidences[i],
                'box': boxes[i]
            })
    
    return detected_objects

def analyze_violations(detected_objects):
    """
    Analyze detected objects to find traffic violations.
    
    Args:
        detected_objects: List of detected objects
        
    Returns:
        violations: List of detected violations
    """
    violations = []
    
    # Count people on motorcycles
    motorcycles = [obj for obj in detected_objects if obj['class'] == 'motorcycle' or obj['class'] == 'motorbike']
    persons = [obj for obj in detected_objects if obj['class'] == 'person']
    
    # Check for triple riding (more than 2 persons on a motorcycle)
    if len(motorcycles) > 0 and len(persons) >= 3:
        # Enhanced proximity checking to ensure persons are on the motorcycle
        for motorcycle in motorcycles:
            m_box = motorcycle['box']
            m_center = (m_box[0] + m_box[2]//2, m_box[1] + m_box[3]//2)
            m_area = m_box[2] * m_box[3]  # Area of the motorcycle bounding box
            
            # Improved counting method
            persons_on_bike = []
            for person in persons:
                p_box = person['box']
                p_center = (p_box[0] + p_box[2]//2, p_box[1] + p_box[3]//2)
                p_area = p_box[2] * p_box[3]  # Area of person bounding box
                
                # Calculate distance between centers, normalized by motorcycle size
                distance = ((p_center[0] - m_center[0])**2 + (p_center[1] - m_center[1])**2)**0.5
                max_distance = max(m_box[2], m_box[3]) * 1.2  # Slightly reduced threshold
                
                # Calculate overlap between bounding boxes
                x_overlap = max(0, min(m_box[0] + m_box[2], p_box[0] + p_box[2]) - max(m_box[0], p_box[0]))
                y_overlap = max(0, min(m_box[1] + m_box[3], p_box[1] + p_box[3]) - max(m_box[1], p_box[1]))
                overlap_area = x_overlap * y_overlap
                
                # Better proximity detection using both distance and overlap
                if distance < max_distance or (overlap_area > 0 and overlap_area / p_area > 0.2):
                    persons_on_bike.append(person)
            
            if len(persons_on_bike) >= 3:
                violations.append("Triple Riding Detected")
                break
    
    # Enhanced helmet violation detection
    if len(motorcycles) > 0 and len(persons) > 0:
        # Check for helmet violations by looking at the upper portion of riders
        for motorcycle in motorcycles:
            m_box = motorcycle['box']
            m_center = (m_box[0] + m_box[2]//2, m_box[1] + m_box[3]//2)
            
            riders = []
            for person in persons:
                p_box = person['box']
                p_center = (p_box[0] + p_box[2]//2, p_box[1] + p_box[3]//2)
                
                # Calculate distance
                distance = ((p_center[0] - m_center[0])**2 + (p_center[1] - m_center[1])**2)**0.5
                
                # If person is close to motorcycle, they are a rider
                if distance < max(m_box[2], m_box[3]) * 1.5:
                    riders.append(person)
            
            # If we found riders, check for helmets using a deterministic approach
            if riders:
                # Check for other objects that might be helmets (we would need a real helmet detector)
                helmets_found = False
                
                # For demonstration purposes, we'll use a more deterministic approach
                # In a real system, you would use a specialized helmet detection model
                for rider in riders:
                    rider_box = rider['box']
                    
                    # Check the upper 1/3 of the rider (where the head/helmet should be)
                    head_height = rider_box[3] // 3
                    head_region = (rider_box[0], rider_box[1], rider_box[2], head_height)
                    
                    # In a real system, this would analyze this region for helmet presence
                    # For demo, we'll check if there's any other non-rider object near the head
                    for obj in detected_objects:
                        if obj not in riders and obj not in motorcycles:
                            obj_box = obj['box']
                            
                            # Check for overlap with head region
                            x_overlap = max(0, min(head_region[0] + head_region[2], obj_box[0] + obj_box[2]) - 
                                          max(head_region[0], obj_box[0]))
                            y_overlap = max(0, min(head_region[1] + head_region[3], obj_box[1] + obj_box[3]) - 
                                          max(head_region[1], obj_box[1]))
                            
                            if x_overlap > 0 and y_overlap > 0:
                                # Found a potential helmet
                                helmets_found = True
                                break
                
                # If not enough helmets for all riders, it's a violation
                if not helmets_found and len(riders) > 0:
                    violations.append("Helmet Violation Detected")
    
    return list(set(violations))  # Remove duplicates

def process_frame(frame, net, output_layers, classes):
    """
    Process a single frame for violations.
    
    Args:
        frame: Input image frame
        net: Pre-loaded YOLO network
        output_layers: Output layer names
        classes: Class names
        
    Returns:
        processed_frame: Frame with detection boxes
        violations: List of detected violations
    """
    # Create a copy of the frame to draw on
    processed_frame = frame.copy()
    
    # Detect objects
    detected_objects = detect_objects(frame, net, output_layers, classes)
    
    # Analyze for violations
    violations = analyze_violations(detected_objects)
    
    # Draw detection boxes
    for obj in detected_objects:
        x, y, w, h = obj['box']
        label = f"{obj['class']} {obj['confidence']:.2f}"
        
        # Set color based on class
        if obj['class'] == 'person':
            color = (0, 255, 0)  # Green for person
        elif obj['class'] == 'motorcycle' or obj['class'] == 'motorbike':
            color = (255, 0, 0)  # Blue for motorcycle
        else:
            color = (0, 0, 255)  # Red for others
        
        # Draw bounding box
        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        cv2.putText(
            processed_frame, 
            label, 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            color, 
            2
        )
    
    # Draw violations
    if violations:
        for i, violation in enumerate(violations):
            cv2.putText(
                processed_frame,
                f"VIOLATION: {violation}",
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red color for violations
                2
            )
    
    return processed_frame, violations

def detect_violations(file_path, is_video=False, original_filename="file"):
    """
    Detect traffic violations in an image or video.
    
    Args:
        file_path: Path to the input file
        is_video: Whether the input is a video
        original_filename: Original filename for output naming
        
    Returns:
        violations: List of detected violations
        output_path: Path to the processed output file
    """
    logger.debug(f"Starting violation detection on: {file_path}")
    
    # Load YOLO model
    net, output_layers, classes = load_yolo()
    
    all_violations = []
    output_path = ""
    
    try:
        if is_video:
            # Process video
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                logger.error(f"Error opening video file: {file_path}")
                return ["Error processing video"], ""
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Create output path
            timestamp = int(time.time())
            output_filename = f"{timestamp}_{original_filename.split('.')[0]}_processed.mp4"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process each frame
            frame_count = 0
            max_frames = 300  # Limit processing to 300 frames (~10 seconds at 30fps)
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every 5th frame to save time
                if frame_count % 5 == 0:
                    processed_frame, violations = process_frame(frame, net, output_layers, classes)
                    all_violations.extend(violations)
                    out.write(processed_frame)
                else:
                    out.write(frame)
                
                frame_count += 1
            
            cap.release()
            out.release()
            
        else:
            # Process image
            frame = cv2.imread(file_path)
            if frame is None:
                logger.error(f"Error reading image file: {file_path}")
                return ["Error processing image"], ""
            
            processed_frame, violations = process_frame(frame, net, output_layers, classes)
            all_violations = violations
            
            # Save processed image
            timestamp = int(time.time())
            output_filename = f"{timestamp}_{original_filename.split('.')[0]}_processed.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cv2.imwrite(output_path, processed_frame)
        
        # Clean up unique violations
        unique_violations = list(set(all_violations))
        logger.debug(f"Detected violations: {unique_violations}")
        
        return unique_violations, output_path
        
    except Exception as e:
        logger.error(f"Error in violation detection: {str(e)}", exc_info=True)
        return [f"Error: {str(e)}"], ""

def save_detection_result(frame, filename, violations=None):
    """Save a processed frame with detection results."""
    if violations:
        for i, violation in enumerate(violations):
            cv2.putText(
                frame,
                f"VIOLATION: {violation}",
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
    
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, frame)
    return output_path
