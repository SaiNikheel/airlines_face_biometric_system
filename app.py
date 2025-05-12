# Import CPU configuration first to disable GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from flask_session import Session
import json
import logging
from google.oauth2 import id_token
from google.auth.transport import requests as greq
import csv
import base64
import numpy as np
import tensorflow as tf

# Configure TensorFlow to use CPU only
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.set_visible_devices([], 'GPU')
            print(f"Disabled GPU device: {device}")
    print("TensorFlow configured to use CPU only")
except Exception as e:
    print(f"Error disabling GPU: {e}")

# Import DeepFace after GPU is disabled
from deepface import DeepFace
import cv2

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
Session(app)

# Google OAuth2 configuration
CLIENT_SECRETS_FILE = "client_secret.json"
SCOPES = ['https://www.googleapis.com/auth/userinfo.profile',
          'https://www.googleapis.com/auth/userinfo.email',
          'openid']

# OAuth2 configuration
REDIRECT_URI = 'http://localhost:3000/oauth2callback'

# Ensure the data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# CSV file path
CSV_FILE = 'data/passenger_data.csv'

# Create CSV file with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['full_name', 'dob', 'passport_id', 'flight_id', 'registration_date', 'face_embedding'])

def is_authenticated():
    return 'user' in session

def get_flow():
    try:
        return Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
    except Exception as e:
        logger.error(f"Error loading client secrets: {str(e)}")
        raise

@app.route('/')
def home():
    if not is_authenticated():
        return render_template('welcome.html')
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if not is_authenticated():
        return redirect(url_for('home'))
    return render_template('welcome.html', user=session['user'])

@app.route('/login')
def login():
    if is_authenticated():
        return redirect(url_for('dashboard'))
    
    try:
        flow = get_flow()
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
        )
        
        session['state'] = state
        logger.debug(f"Generated state: {state}")
        return redirect(authorization_url)
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return render_template('welcome.html', error="Authentication configuration error. Please contact the administrator.")

@app.route('/oauth2callback')
def oauth2callback():
    try:
        # Log the callback parameters
        logger.debug(f"Callback received with state: {request.args.get('state')}")
        logger.debug(f"Session state: {session.get('state')}")
        
        if 'state' not in session:
            raise Exception("No state found in session")
        
        if request.args.get('state') != session['state']:
            raise Exception("State mismatch")
            
        flow = get_flow()
        
        # Get the authorization response from the request
        authorization_response = request.url
        
        # Exchange the authorization code for credentials
        flow.fetch_token(authorization_response=authorization_response)
        
        # Get credentials from the flow
        credentials = flow.credentials
        logger.debug("Successfully obtained credentials")
        
        # Store credentials in session
        session['credentials'] = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
        
        # Get user info
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        
        creds = Credentials(**session['credentials'])
        service = build('oauth2', 'v2', credentials=creds)
        user_info = service.userinfo().get().execute()
        logger.debug(f"Retrieved user info for: {user_info.get('email')}")
        
        session['user'] = {
            'email': user_info.get('email'),
            'name': user_info.get('name'),
            'picture': user_info.get('picture')
        }
        
        return redirect(url_for('dashboard'))
    except Exception as e:
        logger.error(f"Callback error: {str(e)}")
        session.clear()  # Clear the session on error
        return render_template('welcome.html', error=f"Authentication failed: {str(e)}")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/register')
def register_page():
    if not is_authenticated():
        return redirect(url_for('home'))
    return render_template('registration.html')

@app.route('/verify')
def verify_page():
    if not is_authenticated():
        return redirect(url_for('home'))
    return render_template('verification.html')

@app.route('/register', methods=['POST'])
def register_passenger():
    try:
        full_name = request.form['full_name']
        dob = request.form['dob']
        passport_id = request.form['passport_id']
        flight_id = request.form['flight_id']
        face_embedding = json.loads(request.form['face_embedding'])

        # Check if passport ID already exists
        with open(CSV_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['passport_id'] == passport_id:
                    return jsonify({'error': 'Passport ID already registered'}), 400

        # Save to CSV
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([full_name, dob, passport_id, flight_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), json.dumps(face_embedding)])

        return jsonify({'message': 'Registration successful!'})

    except Exception as e:
        logger.error(f"Error in registration: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/verify', methods=['POST'])
def verify():
    if not is_authenticated():
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        passport_id = request.form.get('passport_id')
        
        if not passport_id:
            return jsonify({'error': 'Passport ID is required'}), 400
        
        # Read data from CSV
        df = pd.read_csv(CSV_FILE)
        
        # Find passenger data
        passenger_data = df[df['passport_id'] == passport_id]
        
        if passenger_data.empty:
            return jsonify({'error': 'Passport ID not found'}), 404
        
        # Convert to dictionary
        result = passenger_data.iloc[0].to_dict()
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_face_embedding', methods=['POST'])
def get_face_embedding():
    try:
        logger.info("Starting face embedding generation")
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        try:
            if ',' in data['image']:
                image_data = data['image'].split(',')[1]
            else:
                image_data = data['image']
            image_bytes = base64.b64decode(image_data)
        except Exception as decode_error:
            logger.error(f"Error decoding image: {str(decode_error)}")
            return jsonify({'error': 'Invalid image format. Please provide a valid base64 encoded image.'}), 400
        
        # Save temporary image file
        temp_image_path = 'temp_face.jpg'
        with open(temp_image_path, 'wb') as f:
            f.write(image_bytes)

        logger.info("Image saved to temporary file, starting face detection")

        # First check if a face can be detected (with more relaxed parameters)
        try:
            # Try face detection first with more relaxed settings
            faces = DeepFace.extract_faces(
                img_path=temp_image_path,
                detector_backend='opencv',  # Use OpenCV instead of RetinaFace for faster detection
                enforce_detection=False,    # Don't enforce detection
                align=True
            )
            
            if not faces:
                logger.warning("No face detected in the image")
                os.remove(temp_image_path)
                return jsonify({'error': 'No face detected in the image. Please try again with better lighting and positioning.'}), 400
                
            logger.info(f"Face detection successful. Found {len(faces)} faces.")
            
            # If multiple faces, use the largest one
            if len(faces) > 1:
                logger.warning(f"Multiple faces ({len(faces)}) detected in image. Using the largest one.")
                # Find face with largest area
                largest_face = max(faces, key=lambda x: 
                    x['facial_area']['w'] * x['facial_area']['h'])
                
                # Extract just that face to a new image
                img = cv2.imread(temp_image_path)
                x = largest_face['facial_area']['x']
                y = largest_face['facial_area']['y']
                w = largest_face['facial_area']['w']
                h = largest_face['facial_area']['h']
                
                # Add some margin
                margin = int(max(w, h) * 0.2)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(img.shape[1] - x, w + 2 * margin)
                h = min(img.shape[0] - y, h + 2 * margin)
                
                face_img = img[y:y+h, x:x+w]
                cv2.imwrite(temp_image_path, face_img)
        
        except Exception as detect_error:
            logger.error(f"Error in face detection: {str(detect_error)}")
            # Continue to the embedding generation, as DeepFace.represent has its own detection

        logger.info("Generating face embedding")
        
        # Generate face embedding using DeepFace with CPU-friendly settings
        try:
            embedding = DeepFace.represent(
                img_path=temp_image_path,
                model_name='Facenet',  # Use Facenet instead of Facenet512 for CPU
                detector_backend='opencv',  # Use OpenCV instead of RetinaFace for CPU
                enforce_detection=False,    # More lenient detection
                align=True,
                normalization='base'        # Simpler normalization for CPU
            )
            
            # Clean up temporary file
            os.remove(temp_image_path)
            
            logger.info("Face embedding generated successfully")
            return jsonify({'embedding': embedding[0]['embedding']})
            
        except Exception as embedding_error:
            logger.error(f"Error generating embedding: {str(embedding_error)}")
            
            # Try with even more relaxed settings as a last resort
            try:
                logger.info("Trying with more relaxed settings as fallback")
                embedding = DeepFace.represent(
                    img_path=temp_image_path,
                    model_name='Facenet',
                    detector_backend='skip',  # Skip detection completely
                    enforce_detection=False,
                    align=False              # Skip alignment
                )
                
                # Clean up temporary file
                os.remove(temp_image_path)
                
                logger.info("Face embedding generated with fallback settings")
                return jsonify({'embedding': embedding[0]['embedding']})
                
            except Exception as fallback_error:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                return jsonify({'error': f'Unable to generate face embedding. Please try again with a clearer photo of your face.'}), 400

    except Exception as e:
        if 'temp_face.jpg' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        logger.error(f"Error generating face embedding: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/verify_face', methods=['POST'])
def verify_face():
    try:
        logger.info("Starting face verification")
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        try:
            if ',' in data['image']:
                image_data = data['image'].split(',')[1]
            else:
                image_data = data['image']
            image_bytes = base64.b64decode(image_data)
        except Exception as decode_error:
            logger.error(f"Error decoding image: {str(decode_error)}")
            return jsonify({'error': 'Invalid image format. Please provide a valid base64 encoded image.'}), 400
        
        # Save temporary image file
        temp_image_path = 'temp_verify.jpg'
        with open(temp_image_path, 'wb') as f:
            f.write(image_bytes)

        logger.info("Image saved to temporary file, starting face detection")

        # First, verify face detection quality with more CPU-friendly settings
        try:
            face_objs = DeepFace.extract_faces(
                img_path=temp_image_path,
                detector_backend='opencv',  # Use OpenCV instead of RetinaFace for faster detection
                enforce_detection=False,    # Don't enforce detection
                align=True
            )

            if not face_objs:
                logger.warning("No face detected in the verification image")
                os.remove(temp_image_path)
                return jsonify({'error': 'No face detected in the image. Please try again with better lighting and positioning.'}), 400

            # Check if multiple faces are detected
            if len(face_objs) > 1:
                logger.warning(f"Multiple faces ({len(face_objs)}) detected in the verification image. Using the largest one.")
                # Find face with largest area
                largest_face = max(face_objs, key=lambda x: 
                    x['facial_area']['w'] * x['facial_area']['h'])
                
                face_obj = largest_face
                
                # Extract just that face to a new image
                img = cv2.imread(temp_image_path)
                x = face_obj['facial_area']['x']
                y = face_obj['facial_area']['y']
                w = face_obj['facial_area']['w']
                h = face_obj['facial_area']['h']
                
                # Add some margin
                margin = int(max(w, h) * 0.2)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(img.shape[1] - x, w + 2 * margin)
                h = min(img.shape[0] - y, h + 2 * margin)
                
                face_img = img[y:y+h, x:x+w]
                cv2.imwrite(temp_image_path, face_img)
            else:
                face_obj = face_objs[0]
            
            # Check face size and position with more lenient parameters
            face_area = face_obj['facial_area']
            img = cv2.imread(temp_image_path)
            height, width = img.shape[:2]
            
            # Calculate face size relative to image
            face_width_ratio = face_area['w'] / width
            
            # Face should occupy at least 15% of the image width (more lenient)
            if face_width_ratio < 0.15:
                logger.warning(f"Face too small in verification image. Ratio: {face_width_ratio:.2f}")
                os.remove(temp_image_path)
                return jsonify({
                    'error': 'Face is too small. Please move closer to the camera',
                    'details': f'Face width ratio: {face_width_ratio:.2f}'
                }), 400
                
        except Exception as detect_error:
            logger.error(f"Error in face detection during verification: {str(detect_error)}")
            # Continue anyway, as we'll try with simplified settings

        logger.info("Generating face embedding for verification")
        
        # Generate face embedding with CPU-friendly settings
        try:
            facenet_embedding = DeepFace.represent(
                img_path=temp_image_path,
                model_name='Facenet',  # Use Facenet instead of Facenet512 for CPU
                detector_backend='opencv',  # Use OpenCV instead of RetinaFace for CPU
                enforce_detection=False,    # More lenient detection
                align=True,
                normalization='base'        # Simpler normalization for CPU
            )[0]['embedding']
            
            # We'll skip the VGG-Face model for CPU efficiency
            vgg_embedding = None
            
        except Exception as embedding_error:
            logger.error(f"Error generating verification embedding: {str(embedding_error)}")
            
            # Try with more relaxed settings as fallback
            try:
                logger.info("Trying verification with more relaxed settings")
                facenet_embedding = DeepFace.represent(
                    img_path=temp_image_path,
                    model_name='Facenet',
                    detector_backend='skip',  # Skip detection completely
                    enforce_detection=False,
                    align=False              # Skip alignment
                )[0]['embedding']
                vgg_embedding = None
                
            except Exception as fallback_error:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                logger.error(f"Fallback verification also failed: {str(fallback_error)}")
                return jsonify({'error': 'Unable to process your face for verification. Please try again with a clearer photo.'}), 400

        # Clean up temporary file
        os.remove(temp_image_path)
        logger.info("Face embedding generated for verification")

        # Compare with stored embeddings
        best_match = None
        best_facenet_distance = float('inf')
        best_vgg_distance = float('inf')
        
        # Use more lenient thresholds for CPU-based models
        facenet_threshold = 0.45  # More lenient threshold for cosine similarity
        
        logger.info("Comparing with stored embeddings")
        try:
            with open(CSV_FILE, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stored_embedding = json.loads(row['face_embedding'])
                    
                    # Convert embeddings to numpy arrays
                    facenet_verify_array = np.array(facenet_embedding)
                    stored_array = np.array(stored_embedding)
                    
                    # Calculate cosine similarity for Facenet
                    similarity = np.dot(facenet_verify_array, stored_array) / (np.linalg.norm(facenet_verify_array) * np.linalg.norm(stored_array))
                    facenet_distance = 1 - similarity  # Convert similarity to distance
                    
                    # Track best match
                    if facenet_distance < best_facenet_distance:
                        best_facenet_distance = facenet_distance
                        best_match = row
        except Exception as comparison_error:
            logger.error(f"Error comparing face embeddings: {str(comparison_error)}")
            return jsonify({'error': 'Error comparing face embeddings. Please try again.'}), 500

        logger.debug(f"Best Facenet distance: {best_facenet_distance}, Threshold: {facenet_threshold}")
        
        # Check if we have a match using Facenet only with a more lenient threshold
        if best_match and best_facenet_distance < facenet_threshold:
            logger.info(f"Face verification successful. Match found: {best_match['full_name']}")
            return jsonify({
                'match': True,
                'full_name': best_match['full_name'],
                'dob': best_match['dob'],
                'passport_id': best_match['passport_id'],
                'flight_id': best_match['flight_id'],
                'distance': float(best_facenet_distance),
                'threshold': facenet_threshold
            })
        else:
            logger.info("No matching face found in database")
            return jsonify({
                'match': False, 
                'message': 'No matching face found in database',
                'best_distance': float(best_facenet_distance),
                'threshold': facenet_threshold
            })

    except Exception as e:
        if 'temp_verify.jpg' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        logger.error(f"Error in face verification: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # Only for development
    app.run(host='localhost', port=3000, debug=True) 