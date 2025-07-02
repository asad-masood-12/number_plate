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
    }
</style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown('<h1 class="main-header">🚘 Smart Number Plate Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image or video to detect number plates with AI-powered recognition & time tracking</p>', unsafe_allow_html=True)

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

# === Sidebar Configuration ===
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("🔑 Groq API Key", type="password", help="Enter your Groq API key for enhanced user data generation")
    
    st.markdown("---")
    st.header("📊 Session Stats")
    
    # Display current session stats
    st.metric("Total Detections", st.session_state.session_stats['total_detections'])
    st.metric("Unique Plates", st.session_state.session_stats['unique_plates'])

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
        return model, ocr_reader
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

model, ocr_reader = load_models()

# === Text Cleanup ===
def extract_valid_text(text_list):
    pattern = r'[A-Za-z0-9]+'
    valid_texts = []
    for text in text_list:
        cleaned = "".join(re.findall(pattern, text)).upper()
        if len(cleaned) >= 3:  # Minimum length for valid plate
            valid_texts.append(cleaned)
    return valid_texts

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
            "fuel_cost": "Random amount in dollars ($100-$500)"
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
            # Validate keys
            required_keys = {"owner_name", "phone", "bank_card", "payment_method","fuel_cost"}
            if not required_keys.issubset(user_data.keys()):
                raise ValueError("Groq response missing required fields.")
            return user_data
        else:
            raise ValueError("No valid JSON found in Groq response.")

    except Exception as e:
        st.error(f"❌ Groq API Error: {e}")
        return None

# === Save to MongoDB with In/Out Time Tracking ===
def save_plate_to_db(plate_number):
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
            
            # Determine if this is an "in" or "out" time based on detection count
            if detection_count % 2 == 0:  # Even count = in time
                in_times.append(now)
            else:  # Odd count = out time
                out_times.append(now)
            
            collection.update_one(
                {"plate_number": plate_number},
                {
                    "$inc": {"detection_count": 1}, 
                    "$set": {
                        "last_detection_time": now,
                        "in_times": in_times,
                        "out_times": out_times
                    }
                }
            )
            
            user_info = {
                "owner_name": existing["owner_name"],
                "phone": existing["phone"],
                "bank_card": existing["bank_card"],
                "payment_method": existing["payment_method"],
                "fuel_cost": existing.get("fuel_cost", "N/A"),
                "detection_count": detection_count + 1,
                "in_times": in_times,
                "out_times": out_times,
                "last_detection_time": now,
                "messages": existing.get("messages", [])
            }
        else:
            # Create new plate record
            user_info = generate_dummy_info(plate_number)
            if user_info is None:
                return None  # Groq failed, skip this plate
            
            # First detection is always an "in" time
            user_info.update({
                "plate_number": plate_number,
                "first_detection_time": now,
                "last_detection_time": now,
                "detection_count": 1,
                "in_times": [now],  # First detection is in time
                "out_times": [],     # Empty out times initially
                "messages": []       # Initialize empty messages list
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

# === Image Processing ===
def process_image(image_np):
    if model is None or ocr_reader is None:
        st.error("Models not loaded properly!")
        return image_np, []
    
    detected_plates = []
    results = model(image_np)
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                roi = image_np[y1:y2, x1:x2]
                
                if roi.size > 0:
                    try:
                        ocr_result = ocr_reader.readtext(roi)
                        if ocr_result:
                            texts = [text[1] for text in ocr_result if len(text) > 2 and text[2] > 0.3]  # Confidence threshold
                            plate_texts = extract_valid_text(texts)
                            
                            if plate_texts:
                                plate_text = plate_texts[0]
                                user_info = save_plate_to_db(plate_text)
                                
                                if user_info is not None:
                                    detected_plates.append({
                                        'plate': plate_text,
                                        'user_info': user_info,
                                        'bbox': (x1, y1, x2, y2)
                                    })
                                    
                                    # Draw bounding box and text
                                    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                    cv2.putText(image_np, plate_text, (x1, y1 - 15), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    except Exception as e:
                        st.warning(f"OCR processing error: {e}")
                        continue
    
    return image_np, detected_plates

# === Video Processing ===
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
        
        # Process every 10th frame to speed up
        if frame_count % 10 == 0:
            results = model(frame)
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        roi = frame[y1:y2, x1:x2]
                        
                        if roi.size > 0:
                            try:
                                ocr_result = ocr_reader.readtext(roi)
                                if ocr_result:
                                    texts = [text[1] for text in ocr_result if len(text) > 2 and text[2] > 0.3]
                                    plate_texts = extract_valid_text(texts)
                                    
                                    if plate_texts:
                                        plate_text = plate_texts[0]
                                        user_info = save_plate_to_db(plate_text)
                                        
                                        if user_info is not None:
                                            # Check if this plate is already in detected_plates for this session
                                            existing_plate = next((p for p in detected_plates if p['plate'] == plate_text), None)
                                            if existing_plate:
                                                # Update existing plate info
                                                existing_plate['user_info'] = user_info
                                            else:
                                                # Add new plate
                                                detected_plates.append({
                                                    'plate': plate_text,
                                                    'user_info': user_info
                                                })
                            except Exception as e:
                                continue  # Skip this detection on error
        
        # Draw rectangles for all detected plates in current frame
        results = model(frame)
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "PLATE", (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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

# === Process file only if it's new or not processed yet ===
if uploaded_file is not None and (file_changed or not st.session_state.file_processed):
    file_type = uploaded_file.type
    
    if "image" in file_type:
        # === Image Processing ===
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_np = cv2.imdecode(file_bytes, 1)
        st.session_state.original_image = image_np.copy()
        
        with st.spinner("🔄 Analyzing image..."):
            processed_img, detected_plates = process_image(image_np)
        
        st.session_state.processed_image = processed_img
        st.session_state.current_session_plates = detected_plates
        st.session_state.session_stats['total_detections'] = len(detected_plates)
        st.session_state.session_stats['unique_plates'] = len(set([p['plate'] for p in detected_plates]))
        st.session_state.file_processed = True
    
    elif "video" in file_type:
        # === Video Processing ===
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(uploaded_file.read())
            input_path = temp_input.name
        
        output_path = "processed_video.mp4"
        
        with st.spinner("🔄 Processing video... This may take a while."):
            detected_plates = process_video(input_path, output_path)
        
        st.session_state.processed_video_path = output_path
        st.session_state.current_session_plates = detected_plates[:5]  # Limit to first 5 for videos
        st.session_state.session_stats['total_detections'] = len(detected_plates)
        st.session_state.session_stats['unique_plates'] = len(detected_plates)
        st.session_state.file_processed = True
        
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
                st.image(st.session_state.processed_image, caption="Detected Plates", channels="BGR", use_container_width=True)
                st.success(f"✅ Found {len(st.session_state.current_session_plates)} number plate(s)!")
            else:
                st.warning("⚠️ No number plates detected in this image.")
    
    elif "video" in file_type and st.session_state.processed_video_path:
        st.subheader("🎥 Video Processing")
        
        if st.session_state.current_session_plates:
            st.success(f"✅ Video processed! Found {st.session_state.session_stats['total_detections']} unique plates.")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.video(st.session_state.processed_video_path)
            with col2:
                st.metric("Total Detections", st.session_state.session_stats['total_detections'])
                st.metric("Processing Status", "Complete ✅")
        else:
            st.warning("⚠️ No number plates detected in this video.")

# === Display Current Session Results ===
if st.session_state.current_session_plates:
    st.markdown("---")
    st.subheader("📋 Current Session Results")
    
    # Display detected plates with time tracking and messaging
    for idx, plate_data in enumerate(st.session_state.current_session_plates, 1):
        plate = plate_data['plate']
        plate_info = plate_data['user_info']
        in_times = plate_info.get('in_times', [])
        out_times = plate_info.get('out_times', [])
        detection_count = plate_info.get('detection_count', 1)

        in_times_str = "<br>".join([f"📥 {time}" for time in in_times]) if in_times else "No in times recorded"
        out_times_str = "<br>".join([f"📤 {time}" for time in out_times]) if out_times else "No out times recorded"

        # Display main card
        st.markdown(f"""
        <div class="plate-card">
            <h3>🚗 Plate #{idx}: {plate} (Detected {detection_count} times)</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                <div><strong>👤 Owner:</strong> {plate_info.get('owner_name', 'Unknown')}</div>
                <div><strong>📞 Phone:</strong> {plate_info.get('phone', 'N/A')}</div>
                <div><strong>💳 Bank Card:</strong> {plate_info.get('bank_card', 'N/A')}</div>
                <div><strong>💸 Payment:</strong> {plate_info.get('payment_method', 'N/A')}</div>
                <div><strong>⛽ Fuel Cost:</strong> {plate_info.get('fuel_cost', 'N/A')}</div>
                <div><strong>🕐 Last Seen:</strong> {plate_info.get('last_detection_time', 'N/A')}</div>
                <div><strong>📊 Detection Count:</strong> {plate_info.get('detection_count', 1)}</div>
            </div>
            <div style="margin-top: 15px;">
                <div class="time-log">
                    <strong>📥 IN TIMES:</strong><br>
                    {in_times_str}
                </div>
                <div class="time-log">
                    <strong>📤 OUT TIMES:</strong><br>
                    {out_times_str}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 💬 Message section with simple popup notification
        with st.expander(f"💬 Messages for plate {plate}"):
            # Get fresh messages from database
            messages = get_messages_from_db(plate)
            
            if messages:
                st.write("**Previous Messages:**")
                for i, msg in enumerate(messages, 1):
                    st.markdown(f"{i}. {msg}")
            else:
                st.write("No messages yet.")
            
            # Message input form to prevent rerun on Enter
            with st.form(key=f"message_form_{plate}"):
                new_msg = st.text_input("Type your message:", key=f"msg_input_{plate}")
                submit_button = st.form_submit_button("Send Message")
                
                if submit_button and new_msg.strip():
                    if save_message_to_db(plate, new_msg.strip()):
                        # Create a popup modal overlay
                        st.markdown("""
                        <div id="successPopup" style="
                            position: fixed;
                            top: 0;
                            left: 0;
                            width: 100%;
                            height: 100%;
                            background: rgba(0, 0, 0, 0.5);
                            z-index: 9999;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                        ">
                            <div style="
                                background: white;
                                padding: 30px 40px;
                                border-radius: 15px;
                                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                                text-align: center;
                                max-width: 400px;
                                width: 90%;
                            ">
                                <div style="
                                    font-size: 48px;
                                    color: #28a745;
                                    margin-bottom: 15px;
                                ">✅</div>
                                <h3 style="
                                    color: #333;
                                    margin: 0 0 10px 0;
                                    font-size: 24px;
                                ">Success!</h3>
                                <p style="
                                    color: #666;
                                    margin: 0 0 20px 0;
                                    font-size: 16px;
                                ">Message saved successfully!</p>
                                <button onclick="document.getElementById('successPopup').style.display='none'" style="
                                    background: #28a745;
                                    color: white;
                                    border: none;
                                    padding: 10px 25px;
                                    border-radius: 8px;
                                    cursor: pointer;
                                    font-size: 16px;
                                    font-weight: bold;
                                ">OK</button>
                            </div>
                        </div>
                        
                        <script>
                            // Auto close popup after 3 seconds
                            setTimeout(function() {
                                var popup = document.getElementById('successPopup');
                                if (popup) {
                                    popup.style.display = 'none';
                                }
                            }, 3000);
                        </script>
                        """, unsafe_allow_html=True)
                        
                        # Brief delay before rerun to show the popup
                        import time
                        time.sleep(0.5)
                        st.rerun()  # Only rerun to refresh messages, not reprocess file
                    else:
                        st.error("❌ Failed to save message")
                        st.warning("⚠️ Please check your database connection and try again.")# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🔒 Data with time tracking stored in MongoDB | 🤖 Powered by YOLO & EasyOCR | 🧠 Enhanced by Groq LLaMA</p>
    <p>⏰ <strong>Time Tracking:</strong> Even detections = IN times | Odd detections = OUT times</p>
</div>
""", unsafe_allow_html=True)