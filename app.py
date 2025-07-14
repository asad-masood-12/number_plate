

import streamlit as st
import cv2
import tempfile
import re
import easyocr
import numpy as np
from ultralytics import YOLO
from pymongo import MongoClient
from datetime import datetime
from groq import Groq
import os
import json
import random
import joblib
import pickle

# === Streamlit UI Configuration ===
st.set_page_config(
    page_title="Number Plate Detector", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS for Better UI ===
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .stAlert > div {
        padding: 15px;
        border-radius: 10px;
    }
    .plate-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 5px 0;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        background: #fafafa;
        margin: 20px 0;
    }
    .time-log {
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        max-height: 120px;
        overflow-y: auto;
    }
    .time-entries {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 5px;
    }
    .time-entry {
        background: rgba(255, 255, 255, 0.2);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        white-space: nowrap;
    }
    .feedback-section {
        background: rgba(255, 255, 255, 0.15);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #FFD700;
    }
    .feedback-new {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #333;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .satisfaction-positive {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        color: white;
        border-left: 4px solid #2ECC71;
    }
    .satisfaction-negative {
        background: linear-gradient(135deg, #FF6B6B 0%, #E74C3C 100%);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        color: white;
        border-left: 4px solid #C0392B;
    }
    .discount-section {
        background: linear-gradient(135deg, #FF9A56 0%, #FF6B35 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
        border-left: 4px solid #E55D00;
    }
    .no-discount {
        background: linear-gradient(135deg, #95A5A6 0%, #7F8C8D 100%);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        color: white;
        text-align: center;
    }
    .confidence-info {
        background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        color: white;
        font-size: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown('<h1 class="main-header">🚘 Smart Number Plate Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image or video to detect number plates with AI-powered recognition & time tracking (75% accuracy threshold)</p>', unsafe_allow_html=True)

# === Initialize Session State ===
if 'current_session_plates' not in st.session_state:
    st.session_state.current_session_plates = []
if 'session_stats' not in st.session_state:
    st.session_state.session_stats = {'total_detections': 0, 'unique_plates': 0}
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None
if 'detection_stats' not in st.session_state:
    st.session_state.detection_stats = {'total_attempts': 0, 'successful_extractions': 0, 'low_confidence_rejections': 0}

# === Sidebar Configuration ===
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("🔑 Groq API Key", type="password", help="Enter your Groq API key for enhanced user data generation")
    
    # Detection confidence settings
    st.markdown("---")
    st.header("🎯 Detection Settings")
    yolo_confidence = st.slider("YOLO Detection Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.05, 
                               help="Minimum confidence for YOLO to detect a number plate")
    ocr_confidence = st.slider("OCR Text Confidence", min_value=0.1, max_value=1.0, value=0.75, step=0.05, 
                              help="Minimum confidence for OCR text extraction (75% recommended)")
    
    st.markdown("---")
    st.header("📊 Session Stats")
    
    # Display current session stats
    st.metric("Total Detections", st.session_state.session_stats['total_detections'])
    st.metric("Unique Plates", st.session_state.session_stats['unique_plates'])
    
    # Detection quality stats
    if st.session_state.detection_stats['total_attempts'] > 0:
        success_rate = (st.session_state.detection_stats['successful_extractions'] / 
                       st.session_state.detection_stats['total_attempts']) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
        st.metric("Low Confidence Rejected", st.session_state.detection_stats['low_confidence_rejections'])

# === MongoDB Setup ===
@st.cache_resource
def init_mongodb():
    try:
        MONGO_URI = "mongodb+srv://asadullahmasood1005:o6JMETlQXlGKy8T5@cluster0.nio7sh8.mongodb.net/"
        client = MongoClient(MONGO_URI)
        db = client["car_plate_db"]
        collection = db["plate_records"]
        return collection
    except Exception as e:
        st.error(f"MongoDB connection failed: {e}")
        return None

collection = init_mongodb()

# === Load Models ===
@st.cache_resource
def load_models():
    try:
        MODEL_PATH = "number_plate_best.pt"
        model = YOLO(MODEL_PATH)
        ocr_reader = easyocr.Reader(['en'])
        
        # Load churn prediction model
        try:
            churn_model = joblib.load("churn_model.pkl")
            st.success("✅ Churn prediction model loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Churn model loading failed: {e}")
            churn_model = None
        
        return model, ocr_reader, churn_model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None

model, ocr_reader, churn_model = load_models()

# === Improved Text Cleanup with Pattern Validation ===
def extract_valid_text(text_list, confidence_threshold=0.75):
    """Extract valid number plate text with improved pattern matching and confidence filtering"""
    # Common number plate patterns for different regions
    patterns = [
        r'^[A-Z]{2,3}[0-9]{2,4}[A-Z]?$',        # UK style: AB12CDE
        r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$',    # General: A123BC
        r'^[0-9]{1,3}[A-Z]{1,3}[0-9]{1,4}$',    # Mixed: 123ABC456
        r'^[A-Z]{1,3}[0-9]{3,4}$',              # Simple: ABC1234
        r'^[0-9]{3,4}[A-Z]{2,3}$',              # Reverse: 1234ABC
    ]
    
    valid_texts = []
    
    for text_info in text_list:
        # Extract text and confidence
        if isinstance(text_info, tuple) and len(text_info) >= 3:
            text = text_info[1]
            confidence = text_info[2]
        else:
            text = str(text_info)
            confidence = 1.0  # Default confidence if not provided
        
        # Skip if confidence is below threshold
        if confidence < confidence_threshold:
            st.session_state.detection_stats['low_confidence_rejections'] += 1
            continue
        
        # Clean the text: remove special characters, keep only alphanumeric
        cleaned = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        
        # Check minimum length
        if len(cleaned) < 4:
            continue
        
        # Check maximum length (most plates are under 10 characters)
        if len(cleaned) > 10:
            continue
        
        # Validate against common patterns
        is_valid_pattern = False
        for pattern in patterns:
            if re.match(pattern, cleaned):
                is_valid_pattern = True
                break
        
        # Additional validation: must contain both letters and numbers
        has_letters = bool(re.search(r'[A-Z]', cleaned))
        has_numbers = bool(re.search(r'[0-9]', cleaned))
        
        if is_valid_pattern and has_letters and has_numbers:
            valid_texts.append({
                'text': cleaned,
                'confidence': confidence,
                'original': text
            })
    
    # Sort by confidence and return the best one
    if valid_texts:
        valid_texts.sort(key=lambda x: x['confidence'], reverse=True)
        st.session_state.detection_stats['successful_extractions'] += 1
        return [valid_texts[0]['text']]  # Return only the highest confidence result
    
    return []

# === Calculate Discounts ===
def calculate_discounts(fuel_cost_str, has_loyalty_card, detection_count):
    """Calculate discounts based on loyalty card and visit count"""
    try:
        # Extract numeric value from fuel cost string
        import re
        original_cost = float(re.findall(r'\d+', fuel_cost_str)[0])
        
        total_discount_percent = 0
        discount_details = []
        
        # 5% off for loyalty card
        if has_loyalty_card:
            total_discount_percent += 5
            discount_details.append("5% Loyalty Card Discount")
        
        # 10% off for 10th visit
        if detection_count >= 10:
            total_discount_percent += 10
            discount_details.append("10% 10th Visit Discount")
        
        # Calculate final cost
        discount_amount = original_cost * (total_discount_percent / 100)
        final_cost = original_cost - discount_amount
        
        return {
            "original_cost": original_cost,
            "discount_percent": total_discount_percent,
            "discount_amount": discount_amount,
            "final_cost": final_cost,
            "discount_details": discount_details,
            "has_discounts": len(discount_details) > 0
        }
    except Exception as e:
        return None

def predict_customer_satisfaction(fuel_cost_str):
    """Predict customer satisfaction based on fuel cost using churn model"""
    if churn_model is None:
        return None, "Model not available"
    
    try:
        # Extract numeric value from fuel cost string (e.g., "$150" -> 150)
        import re
        fuel_cost_numeric = float(re.findall(r'\d+', fuel_cost_str)[0])
        
        # Predict using the churn model
        prediction = churn_model.predict([[fuel_cost_numeric]])[0]
        prediction_proba = churn_model.predict_proba([[fuel_cost_numeric]])[0]
        
        # Determine satisfaction based on prediction
        if prediction == 0 or prediction_proba[1] < 0.5:
            satisfaction_status = "😞 Not Satisfied"
            satisfaction_message = "Customer may be dissatisfied with fuel costs"
            satisfaction_color = "#FF6B6B"
        else:
            satisfaction_status = "😊 Happy Customer"
            satisfaction_message = "Customer is satisfied with fuel costs"
            satisfaction_color = "#4ECDC4"
        
        return {
            "status": satisfaction_status,
            "message": satisfaction_message,
            "color": satisfaction_color,
            "prediction": int(prediction),
            "probability": float(prediction_proba[1]),
            "fuel_cost_numeric": fuel_cost_numeric
        }, None
        
    except Exception as e:
        return None, f"Prediction error: {e}"

def generate_random_feedback():
    """Generate random customer feedback for gas station service"""
    feedbacks = [
        "⛽ Great service! The staff was friendly and quick.",
        "👍 Everything was clean and well-organized. Keep it up!",
        "😊 I love how fast the fueling process is. Very efficient!",
        "🧼 The station was tidy, but the restroom could use some cleaning.",
        "🕒 The wait time was a bit long during peak hours. Maybe add more staff?",
        "🔥 Excellent service and great attitude from the attendants!",
        "🧃 I appreciate the mini-mart having fresh snacks and drinks.",
        "💡 Maybe you could add more shade near the pumps. It gets hot in summer!",
        "📱 Loved the mobile payment option — super convenient!",
        "🛠️ The air pump was out of order. Hope it gets fixed soon.",
        "😃 Fuel quality seems good and my car runs smooth after refueling.",
        "💬 A suggestion: maybe install a small coffee stand for early commuters.",
        "🌟 Staff were really helpful — even cleaned my windshield!",
        "🪑 It would be nice to have a small waiting area or bench.",
        "📶 Wi-Fi was a great touch. Helped pass the time while fueling."
    ]
    
    return random.choice(feedbacks)

# === Generate User Info with Groq ===
def generate_dummy_info(plate_number):
    try:
        # Use the API key from the sidebar input
        if not groq_api_key:
            st.warning("⚠️ Please enter your Groq API key in the sidebar")
            return None
            
        client = Groq(api_key=groq_api_key)
        prompt = f"""
        Generate realistic Pakistani user data for car plate {plate_number}.
        for every car plate detection it should be unique owner_name
        for every car plate detection it should be unique Phone number
        for every car plate detection it should be unique back card number
        for every car plate detection it can by any one of payment method EasyPaisa or JazzCash or Credit Card,
        for every detected car plate, assign a random fuel cost representing how much the car spent on fuel.

        Return ONLY valid JSON with these exact keys:
        {{
            "owner_name": "Pakistani name",
            "phone": "03XXXXXXXXX format",
            "bank_card": "**** **** **** XXXX",
            "payment_method": "EasyPaisa or JazzCash or Credit Card",
            "fuel_cost": "Random amount in dollars ($1-$100) (make it different each time)"
        }}
        """
        
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You generate Pakistani user data in JSON format only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        reply = chat_completion.choices[0].message.content.strip()
        json_start = reply.find('{')
        json_end = reply.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = reply[json_start:json_end]
            user_data = json.loads(json_str)
            # Validate keys and add fallbacks for missing fields
            required_keys = {"owner_name", "phone", "bank_card", "payment_method","fuel_cost"}
            if not required_keys.issubset(user_data.keys()):
                raise ValueError("Groq response missing required fields.")
            
            fuel_cost_num = 0
            if "fuel_cost" in user_data:
                try:
                    fuel_cost_num = float(re.findall(r'\d+', user_data["fuel_cost"])[0])
                except:
                    fuel_cost_num = 0

            # If fuel cost is missing, 0, or around 43 (Groq's favorite number), generate random
            if "fuel_cost" not in user_data or fuel_cost_num == 0 or 40 <= fuel_cost_num <= 50:
                random_cost = random.randint(1, 100)
                user_data["fuel_cost"] = f"${random_cost}"
            
            # Add loyalty_card if missing (50% chance of having loyalty card)
            if "loyalty_card" not in user_data:
                user_data["loyalty_card"] = random.choice([True, False])
            
            # Ensure loyalty_card is boolean
            if isinstance(user_data["loyalty_card"], str):
                user_data["loyalty_card"] = user_data["loyalty_card"].lower() == "true"
            
            return user_data
        else:
            raise ValueError("No valid JSON found in Groq response.")

    except Exception as e:
        st.error(f"❌ Groq API Error: {e}")
        return None

# === Save to MongoDB with In/Out Time Tracking and Auto Feedback ===
def save_plate_to_db(plate_number, confidence_score=None):
    if collection is None:
        return None
    
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        existing = collection.find_one({"plate_number": plate_number})
        
        if existing:
            # Update existing plate with new detection time
            in_times = existing.get("in_times", [])
            out_times = existing.get("out_times", [])
            detection_count = existing.get("detection_count", 0)
            
            # Calculate discounts based on loyalty card and visit count
            discount_info = calculate_discounts(
                existing.get("fuel_cost", "$0"), 
                existing.get("loyalty_card", False), 
                detection_count + 1  # Include current detection
            )
            
            # Determine if this is an "in" or "out" time based on detection count
            if detection_count % 2 == 0:  # Even count = in time
                in_times.append(now)
            else:  # Odd count = out time
                out_times.append(now)
            
            # Generate automatic feedback if detection count is less than 2
            auto_feedback = None
            satisfaction_data = None
            if detection_count < 2:
                auto_feedback = generate_random_feedback()
                # Predict customer satisfaction
                satisfaction_data, error = predict_customer_satisfaction(existing.get("fuel_cost", "$0"))
                
                # Add feedback to messages
                existing_messages = existing.get("messages", [])
                existing_messages.append(f"[AUTO-FEEDBACK] {auto_feedback}")
                
                collection.update_one(
                    {"plate_number": plate_number},
                    {
                        "$inc": {"detection_count": 1}, 
                        "$set": {
                            "last_detection_time": now,
                            "in_times": in_times,
                            "out_times": out_times,
                            "messages": existing_messages,
                            "auto_feedback": auto_feedback,
                            "feedback_generated": True,
                            "satisfaction_prediction": satisfaction_data if satisfaction_data else None,
                            "discount_info": discount_info,
                            "last_confidence_score": confidence_score
                        }
                    }
                )
            else:
                collection.update_one(
                    {"plate_number": plate_number},
                    {
                        "$inc": {"detection_count": 1}, 
                        "$set": {
                            "last_detection_time": now,
                            "in_times": in_times,
                            "out_times": out_times,
                            "discount_info": discount_info,
                            "last_confidence_score": confidence_score
                        }
                    }
                )
            
            user_info = {
                "owner_name": existing["owner_name"],
                "phone": existing["phone"],
                "bank_card": existing["bank_card"],
                "payment_method": existing["payment_method"],
                "fuel_cost": existing.get("fuel_cost", "N/A"),
                "loyalty_card": existing.get("loyalty_card", False),
                "detection_count": detection_count + 1,
                "in_times": in_times,
                "out_times": out_times,
                "last_detection_time": now,
                "messages": existing.get("messages", []),
                "auto_feedback": auto_feedback,
                "feedback_generated": existing.get("feedback_generated", False) or (detection_count < 2),
                "satisfaction_prediction": satisfaction_data if satisfaction_data else existing.get("satisfaction_prediction"),
                "discount_info": discount_info,
                "confidence_score": confidence_score
            }
        else:
            # Create new plate record
            user_info = generate_dummy_info(plate_number)
            if user_info is None:
                return None  # Groq failed, skip this plate
            
            # Generate automatic feedback for new plate (detection count = 1)
            auto_feedback = generate_random_feedback()
            
            # Calculate discounts based on loyalty card and visit count
            discount_info = calculate_discounts(
                user_info.get("fuel_cost", "$0"), 
                user_info.get("loyalty_card", False), 
                1  # First visit
            )
            
            # Predict customer satisfaction based on final fuel cost (after discount)
            final_cost_str = f"${discount_info['final_cost']:.0f}" if discount_info else user_info.get("fuel_cost", "$0")
            satisfaction_data, error = predict_customer_satisfaction(final_cost_str)
            
            # First detection is always an "in" time
            user_info.update({
                "plate_number": plate_number,
                "first_detection_time": now,
                "last_detection_time": now,
                "detection_count": 1,
                "in_times": [now],  # First detection is in time
                "out_times": [],     # Empty out times initially
                "messages": [f"[AUTO-FEEDBACK] {auto_feedback}"],  # Auto feedback as first message
                "auto_feedback": auto_feedback,
                "feedback_generated": True,
                "satisfaction_prediction": satisfaction_data if satisfaction_data else None,
                "discount_info": discount_info,
                "confidence_score": confidence_score
            })
            collection.insert_one(user_info)
        
        return user_info
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        return None

# === Function to save message to database ===
def save_message_to_db(plate_number, message):
    if collection is None:
        return False
    
    try:
        collection.update_one(
            {"plate_number": plate_number},
            {"$push": {"messages": message}}
        )
        return True
    except Exception as e:
        st.error(f"Failed to save message: {e}")
        return False

# === Function to get messages from database ===
def get_messages_from_db(plate_number):
    if collection is None:
        return []
    
    try:
        record = collection.find_one({"plate_number": plate_number})
        if record:
            return record.get("messages", [])
        return []
    except Exception as e:
        st.error(f"Failed to get messages: {e}")
        return []

# === Improved Image Processing with Enhanced Confidence Filtering ===
def process_image(image_np):
    if model is None or ocr_reader is None:
        st.error("Models not loaded properly!")
        return image_np, []
    
    detected_plates = []
    
    # Use YOLO with confidence threshold from sidebar
    results = model(image_np, conf=yolo_confidence)
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()  # Get detection confidences
            
            for box, detection_conf in zip(boxes, confidences):
                # Increment total attempts
                st.session_state.detection_stats['total_attempts'] += 1
                
                x1, y1, x2, y2 = map(int, box)
                roi = image_np[y1:y2, x1:x2]
                
                if roi.size > 0:
                    try:
                        # Preprocess ROI for better OCR
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        
                        # Apply image preprocessing to improve OCR accuracy
                        # Resize if too small
                        height, width = roi_gray.shape
                        if height < 50 or width < 150:
                            scale_factor = max(50/height, 150/width)
                            new_width = int(width * scale_factor)
                            new_height = int(height * scale_factor)
                            roi_gray = cv2.resize(roi_gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                        
                        # Apply adaptive thresholding
                        roi_thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        
                        # Use both original and preprocessed images for OCR
                        ocr_result1 = ocr_reader.readtext(roi)
                        ocr_result2 = ocr_reader.readtext(roi_thresh)
                        
                        # Combine results
                        all_ocr_results = ocr_result1 + ocr_result2
                        
                        if all_ocr_results:
                            # Use improved text extraction with confidence filtering
                            plate_texts = extract_valid_text(all_ocr_results, confidence_threshold=ocr_confidence)
                            
                            if plate_texts:
                                plate_text = plate_texts[0]
                                
                                # Get the confidence score from OCR results
                                best_confidence = 0
                                for result in all_ocr_results:
                                    if len(result) >= 3 and result[1].upper().replace(' ', '') == plate_text.replace(' ', ''):
                                        best_confidence = max(best_confidence, result[2])
                                
                                # Save to database with confidence score
                                user_info = save_plate_to_db(plate_text, best_confidence)
                                
                                if user_info is not None:
                                    detected_plates.append({
                                        'plate': plate_text,
                                        'user_info': user_info,
                                        'bbox': (x1, y1, x2, y2),
                                        'yolo_confidence': float(detection_conf),
                                        'ocr_confidence': float(best_confidence)
                                    })
                                    
                                    # Draw bounding box and text with confidence info
                                    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                    cv2.putText(image_np, f"{plate_text}", (x1, y1 - 30), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                    cv2.putText(image_np, f"YOLO: {detection_conf:.2f} OCR: {best_confidence:.2f}", 
                                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            else:
                                # Draw rejected detection in red
                                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(image_np, "LOW CONFIDENCE", (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    except Exception as e:
                        st.warning(f"OCR processing error: {e}")
                        continue
    
    return image_np, detected_plates

# === Improved Video Processing with Enhanced Confidence Filtering ===
def process_video(input_path, output_path):
    if model is None or ocr_reader is None:
        st.error("Models not loaded properly!")
        return []
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    detected_plates = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        # Process every 30th frame to speed up
        if frame_count % 30 == 0:
            results = model(frame, conf=yolo_confidence)
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, detection_conf in zip(boxes, confidences):
                        st.session_state.detection_stats['total_attempts'] += 1
                        
                        x1, y1, x2, y2 = map(int, box)
                        roi = frame[y1:y2, x1:x2]
                        
                        if roi.size > 0:
                            try:
                                # Preprocess ROI for better OCR
                                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                
                                # Apply image preprocessing
                                height_roi, width_roi = roi_gray.shape
                                if height_roi < 50 or width_roi < 150:
                                    scale_factor = max(50/height_roi, 150/width_roi)
                                    new_width = int(width_roi * scale_factor)
                                    new_height = int(height_roi * scale_factor)
                                    roi_gray = cv2.resize(roi_gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                                
                                roi_thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                                
                                # Use both original and preprocessed images for OCR
                                ocr_result1 = ocr_reader.readtext(roi)
                                ocr_result2 = ocr_reader.readtext(roi_thresh)
                                
                                all_ocr_results = ocr_result1 + ocr_result2
                                
                                if all_ocr_results:
                                    plate_texts = extract_valid_text(all_ocr_results, confidence_threshold=ocr_confidence)
                                    
                                    if plate_texts:
                                        plate_text = plate_texts[0]
                                        
                                        # Get the confidence score
                                        best_confidence = 0
                                        for result in all_ocr_results:
                                            if len(result) >= 3 and result[1].upper().replace(' ', '') == plate_text.replace(' ', ''):
                                                best_confidence = max(best_confidence, result[2])
                                        
                                        user_info = save_plate_to_db(plate_text, best_confidence)
                                        
                                        if user_info is not None:
                                            # Check if this plate is already in detected_plates for this session
                                            existing_plate = next((p for p in detected_plates if p['plate'] == plate_text), None)
                                            if existing_plate:
                                                # Update existing plate info with better confidence if available
                                                if best_confidence > existing_plate.get('ocr_confidence', 0):
                                                    existing_plate['user_info'] = user_info
                                                    existing_plate['ocr_confidence'] = best_confidence
                                                    existing_plate['yolo_confidence'] = max(existing_plate.get('yolo_confidence', 0), detection_conf)
                                            else:
                                                # Add new plate
                                                detected_plates.append({
                                                    'plate': plate_text,
                                                    'user_info': user_info,
                                                    'yolo_confidence': float(detection_conf),
                                                    'ocr_confidence': float(best_confidence)
                                                })
                            except Exception as e:
                                continue  # Skip this detection on error
        
        # Draw rectangles for all detections in current frame
        results = model(frame, conf=yolo_confidence)
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for box, detection_conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Color based on confidence
                    if detection_conf >= yolo_confidence:
                        color = (0, 255, 0)  # Green for good detection
                        label = f"PLATE {detection_conf:.2f}"
                    else:
                        color = (0, 165, 255)  # Orange for lower confidence
                        label = f"WEAK {detection_conf:.2f}"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        out.write(frame)
    
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    return detected_plates

# === Main Upload Section ===
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📁 Choose an Image or Video File", 
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    help="Supported formats: JPG, JPEG, PNG for images | MP4, AVI, MOV for videos"
)
st.markdown('</div>', unsafe_allow_html=True)

# === Check if a new file was uploaded ===
file_changed = False
if uploaded_file is not None:
    # Create a unique identifier for the uploaded file
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    if st.session_state.last_uploaded_file != file_id:
        file_changed = True
        st.session_state.last_uploaded_file = file_id
        st.session_state.file_processed = False
        # Clear previous session results
        st.session_state.current_session_plates = []
        st.session_state.session_stats = {'total_detections': 0, 'unique_plates': 0}
        st.session_state.processed_image = None
        st.session_state.original_image = None
        st.session_state.processed_video_path = None
        # Reset detection stats for new file
        st.session_state.detection_stats = {'total_attempts': 0, 'successful_extractions': 0, 'low_confidence_rejections': 0}

# === Process file only if it's new or not processed yet ===
if uploaded_file is not None and (file_changed or not st.session_state.file_processed):
    file_type = uploaded_file.type
    
    if "image" in file_type:
        # === Image Processing ===
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_np = cv2.imdecode(file_bytes, 1)
        st.session_state.original_image = image_np.copy()
        
        with st.spinner("🔄 Analyzing image with enhanced confidence filtering..."):
            processed_img, detected_plates = process_image(image_np)
        
        st.session_state.processed_image = processed_img
        st.session_state.current_session_plates = detected_plates
        st.session_state.session_stats['total_detections'] = len(detected_plates)
        st.session_state.session_stats['unique_plates'] = len(set([p['plate'] for p in detected_plates]))
        st.session_state.file_processed = True
        
        # Display confidence filtering results
        if st.session_state.detection_stats['total_attempts'] > 0:
            st.markdown(f"""
            <div class="confidence-info">
                📊 <strong>Detection Quality Report:</strong><br>
                Total Attempts: {st.session_state.detection_stats['total_attempts']} | 
                Successful: {st.session_state.detection_stats['successful_extractions']} | 
                Rejected (Low Confidence): {st.session_state.detection_stats['low_confidence_rejections']}
            </div>
            """, unsafe_allow_html=True)
    
    elif "video" in file_type:
        # === Video Processing ===
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(uploaded_file.read())
            input_path = temp_input.name
        
        output_path = "processed_video.mp4"
        
        with st.spinner("🔄 Processing video with enhanced confidence filtering... This may take a while."):
            detected_plates = process_video(input_path, output_path)
        
        st.session_state.processed_video_path = output_path
        st.session_state.current_session_plates = detected_plates[:10]  # Limit to first 10 for videos
        st.session_state.session_stats['total_detections'] = len(detected_plates)
        st.session_state.session_stats['unique_plates'] = len(detected_plates)
        st.session_state.file_processed = True
        
        # Display confidence filtering results for video
        if st.session_state.detection_stats['total_attempts'] > 0:
            st.markdown(f"""
            <div class="confidence-info">
                📊 <strong>Video Detection Quality Report:</strong><br>
                Total Attempts: {st.session_state.detection_stats['total_attempts']} | 
                Successful: {st.session_state.detection_stats['successful_extractions']} | 
                Rejected (Low Confidence): {st.session_state.detection_stats['low_confidence_rejections']}
            </div>
            """, unsafe_allow_html=True)
        
        # Clean up temp file
        try:
            os.unlink(input_path)
        except:
            pass

# === Display results if available ===
if uploaded_file is not None and st.session_state.file_processed:
    file_type = uploaded_file.type
    
    if "image" in file_type and st.session_state.original_image is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(st.session_state.original_image, caption="Uploaded Image", channels="BGR", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Processing Results")
            if st.session_state.current_session_plates:
                st.image(st.session_state.processed_image, caption="Detected Plates (Green=Valid, Red=Rejected)", channels="BGR", use_container_width=True)
                st.success(f"✅ Found {len(st.session_state.current_session_plates)} valid number plate(s) above {ocr_confidence*100:.0f}% confidence!")
            else:
                st.warning(f"⚠️ No number plates detected above {ocr_confidence*100:.0f}% confidence threshold.")
                if st.session_state.detection_stats['low_confidence_rejections'] > 0:
                    st.info(f"ℹ️ {st.session_state.detection_stats['low_confidence_rejections']} plates were detected but rejected due to low confidence. Try lowering the OCR confidence threshold in the sidebar.")
    
    elif "video" in file_type and st.session_state.processed_video_path:
        st.subheader("🎥 Video Processing")
        
        if st.session_state.current_session_plates:
            st.success(f"✅ Video processed! Found {st.session_state.session_stats['total_detections']} plates above {ocr_confidence*100:.0f}% confidence.")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.video(st.session_state.processed_video_path)
            with col2:
                st.metric("Valid Detections", st.session_state.session_stats['total_detections'])
                st.metric("Confidence Threshold", f"{ocr_confidence*100:.0f}%")
                st.metric("Processing Status", "Complete ✅")
        else:
            st.warning(f"⚠️ No number plates detected above {ocr_confidence*100:.0f}% confidence threshold.")
            if st.session_state.detection_stats['low_confidence_rejections'] > 0:
                st.info(f"ℹ️ {st.session_state.detection_stats['low_confidence_rejections']} plates were detected but rejected due to low confidence.")

# === Display Current Session Results ===
if st.session_state.current_session_plates:
    st.markdown("---")
    st.subheader("📋 Current Session Results (High Confidence Only)")
    
    # Display detected plates with confidence scores and enhanced information
    for idx, plate_data in enumerate(st.session_state.current_session_plates, 1):
        plate = plate_data['plate']
        plate_info = plate_data['user_info']
        in_times = plate_info.get('in_times', [])
        out_times = plate_info.get('out_times', [])
        detection_count = plate_info.get('detection_count', 1)
        auto_feedback = plate_info.get('auto_feedback', '')
        feedback_generated = plate_info.get('feedback_generated', False)
        satisfaction_prediction = plate_info.get('satisfaction_prediction', None)
        discount_info = plate_info.get('discount_info', None)
        loyalty_card = plate_info.get('loyalty_card', False)
        yolo_conf = plate_data.get('yolo_confidence', 0)
        ocr_conf = plate_data.get('ocr_confidence', 0)

        # Format times for horizontal display
        in_times_html = ""
        if in_times:
            in_entries = [f'<span class="time-entry">📥 {time}</span>' for time in in_times]
            in_times_html = f'<div class="time-entries">{"".join(in_entries)}</div>'
        else:
            in_times_html = '<p style="margin: 5px 0; font-style: italic;">No in times recorded</p>'

        out_times_html = ""
        if out_times:
            out_entries = [f'<span class="time-entry">📤 {time}</span>' for time in out_times]
            out_times_html = f'<div class="time-entries">{"".join(out_entries)}</div>'
        else:
            out_times_html = '<p style="margin: 5px 0; font-style: italic;">No out times recorded</p>'

        # Display main card with confidence scores
        st.markdown(f"""
        <div class="plate-card">
            <h3>🚗 Plate #{idx}: {plate} (Detected {detection_count} times)</h3>
            <div class="confidence-info" style="background: rgba(255,255,255,0.2); margin: 10px 0; padding: 8px; border-radius: 5px;">
                🎯 <strong>Detection Quality:</strong> YOLO Confidence: {yolo_conf:.1%} | OCR Confidence: {ocr_conf:.1%}
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                <div><strong>👤 Owner:</strong> {plate_info.get('owner_name', 'Unknown')}</div>
                <div><strong>📞 Phone:</strong> {plate_info.get('phone', 'N/A')}</div>
                <div><strong>💳 Bank Card:</strong> {plate_info.get('bank_card', 'N/A')}</div>
                <div><strong>💸 Payment:</strong> {plate_info.get('payment_method', 'N/A')}</div>
                <div><strong>⛽ Fuel Cost:</strong> {plate_info.get('fuel_cost', 'N/A')}</div>
                <div><strong>💎 Loyalty Card:</strong> {'✅ Yes' if loyalty_card else '❌ No'}</div>
                <div><strong>🕐 Last Seen:</strong> {plate_info.get('last_detection_time', 'N/A')}</div>
                <div><strong>📊 Detection Count:</strong> {plate_info.get('detection_count', 1)}</div>
            </div>
            <div style="margin-top: 15px;">
                <div class="time-log">
                    <strong>📥 IN TIMES:</strong>
                    {in_times_html}
                </div>
                <div class="time-log">
                    <strong>📤 OUT TIMES:</strong>
                    {out_times_html}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 💰 Discount Section
        if discount_info and discount_info['has_discounts']:
            st.markdown(f"""
            <div class="discount-section">
                <h4>💰 Discounts Applied:</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 10px 0;">
                    <div><strong>Original Cost:</strong> ${discount_info['original_cost']:.2f}</div>
                    <div><strong>Total Discount:</strong> {discount_info['discount_percent']}% (-${discount_info['discount_amount']:.2f})</div>
                    <div><strong>Final Cost:</strong> ${discount_info['final_cost']:.2f}</div>
                    <div><strong>You Saved:</strong> ${discount_info['discount_amount']:.2f}</div>
                </div>
                <p style="margin: 5px 0; font-size: 14px;">
                    🎯 <strong>Discounts:</strong> {' + '.join(discount_info['discount_details'])}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="no-discount">
                <p style="margin: 5px 0;">💸 No discounts available for this visit</p>
                <small>💡 Get a loyalty card for 5% off or visit us 10 times for 10% off!</small>
            </div>
            """, unsafe_allow_html=True)
        
        # AI Satisfaction Prediction
        if satisfaction_prediction:
            satisfaction_class = "satisfaction-positive" if satisfaction_prediction['prediction'] == 1 else "satisfaction-negative"
            st.markdown(f"""
            <div class="{satisfaction_class}">
                <h4>🧠 AI Customer Satisfaction Prediction:</h4>
                <p style="margin: 5px 0; font-size: 16px;"><strong>{satisfaction_prediction['status']}</strong></p>
                <p style="margin: 5px 0; font-size: 14px;">{satisfaction_prediction['message']}</p>
                <small>Based on fuel cost: ${satisfaction_prediction['fuel_cost_numeric']}</small>
            </div>
            """, unsafe_allow_html=True)

        # 🌟 Auto Feedback Section
        if auto_feedback and feedback_generated:
            feedback_class = "feedback-new" if detection_count <= 2 else "feedback-section"
            st.markdown(f"""
            <div class="{feedback_class}">
                <h4>🌟 Customer Feedback:</h4>
                <p style="margin: 5px 0; font-size: 16px;">{auto_feedback}</p>
                <small>{'' if detection_count <= 2 else '📝 Previous feedback'}</small>
            </div>
            """, unsafe_allow_html=True)

        # 💬 Message section
        with st.expander(f"💬 Messages for plate {plate}"):
            # Get fresh messages from database
            messages = get_messages_from_db(plate)
            
            if messages:
                st.write("**All Messages:**")
                for i, msg in enumerate(messages, 1):
                    if msg.startswith("[AUTO-FEEDBACK]"):
                        st.markdown(f"🤖 **{i}.** {msg.replace('[AUTO-FEEDBACK]', '')}")
                    else:
                        st.markdown(f"👤 **{i}.** {msg}")
            else:
                st.write("No messages yet.")
            
            # Message input form
            with st.form(key=f"message_form_{plate}"):
                new_msg = st.text_input("Type your message:", key=f"msg_input_{plate}")
                submit_button = st.form_submit_button("Send Message")
                
                if submit_button and new_msg.strip():
                    if save_message_to_db(plate, new_msg.strip()):
                        st.success("✅ Message saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save message")

# === Footer ===
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🔒 Enhanced data with confidence filtering | 🤖 Powered by YOLO & EasyOCR | 🧠 Enhanced by Groq LLaMA</p>
    <p>🎯 <strong>Quality Control:</strong> OCR Confidence ≥ {ocr_confidence*100:.0f}% | YOLO Confidence ≥ {yolo_confidence*100:.0f}%</p>
    <p>⏰ <strong>Time Tracking:</strong> Even detections = IN times | Odd detections = OUT times</p>
    <p>🌟 <strong>Auto Feedback:</strong> Automatic welcome messages for new vehicles (detection count ≤ 2)</p>
    <p>🧠 <strong>AI Satisfaction:</strong> Churn prediction model analyzes fuel costs to predict customer satisfaction</p>
    <p>💰 <strong>Discount System:</strong> 5% off with loyalty card | 10% off on 10th visit</p>
</div>
""", unsafe_allow_html=True)
