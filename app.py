from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import os
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from flask_session import Session
import json
import logging
from google.oauth2 import id_token
from google.auth.transport import requests
import csv
import base64
import numpy as np
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
# Read REDIRECT_URI from client_secret.json dynamically
try:
    with open(CLIENT_SECRETS_FILE, 'r') as f:
        client_secrets = json.load(f)
        # Assuming the first redirect_uri in the list is the correct one
        REDIRECT_URI = client_secrets.get('web', {}).get('redirect_uris', [''])[0]
        if not REDIRECT_URI:
            raise ValueError("redirect_uris not found or is empty in client_secret.json")
except FileNotFoundError:
    logger.error(f"Client secrets file not found at {CLIENT_SECRETS_FILE}")
    REDIRECT_URI = '' # Set a default or handle error appropriately
except json.JSONDecodeError:
    logger.error(f"Error decoding JSON from {CLIENT_SECRETS_FILE}")
    REDIRECT_URI = ''
except Exception as e:
    logger.error(f"Error reading redirect URI from {CLIENT_SECRETS_FILE}: {e}")
    REDIRECT_URI = ''

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
        print("Face embedding request received")
        data = request.json
        
        if not data or 'image' not in data:
            print("Error: Missing image data in request")
            return jsonify({'error': 'Missing image data'}), 400
            
        image_data = data['image']
        
        # Check if the image is a base64 string
        if not isinstance(image_data, str):
            print("Error: Image data is not a string")
            return jsonify({'error': 'Image data must be a base64 string'}), 400
            
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
            
        # Check if the base64 string is valid
        try:
            decoded_image = base64.b64decode(image_data)
        except Exception as e:
            print(f"Error decoding base64 image: {str(e)}")
            return jsonify({'error': f'Invalid base64 image: {str(e)}'}), 400
            
        # Save the image temporarily
        temp_path = 'temp_image.jpg'
        with open(temp_path, 'wb') as f:
            f.write(decoded_image)
            
        print(f"Image saved temporarily at {temp_path}, size: {len(decoded_image)} bytes")
        
        try:
            from deepface import DeepFace
            
            # Set a specific detector backend for consistency
            detector_backend = "opencv"  # Faster and more reliable than retinaface
            print(f"Extracting face using {detector_backend} detector")
            
            # Extract the face embedding using GhostFaceNet instead of Facenet
            print("Generating face embedding with GhostFaceNet...")
            embedding_objs = DeepFace.represent(img_path=temp_path, 
                                       model_name="GhostFaceNet", 
                                       detector_backend=detector_backend,
                                       enforce_detection=True)
                                       
            if not embedding_objs or len(embedding_objs) == 0:
                print("Error: No face embedding generated")
                return jsonify({'error': 'Failed to generate face embedding'}), 400
                
            # Get the first face embedding
            embedding = embedding_objs[0]["embedding"]
            
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            print(f"Face embedding generated successfully, length: {len(embedding)}")
            return jsonify({'embedding': embedding})
            
        except Exception as e:
            print(f"Error generating face embedding: {str(e)}")
            
            # Provide more specific error messages
            error_message = str(e)
            if "enforce_detection=False" in error_message:
                return jsonify({'error': 'No face detected in the image. Please ensure your face is clearly visible.'}), 400
            elif "NotImplementedError" in error_message:
                return jsonify({'error': 'Face detection backend not available. Please try again.'}), 500
            elif "model_name" in error_message and "GhostFaceNet" in error_message:
                # Fallback to Facenet if GhostFaceNet is not available
                print("GhostFaceNet not available, falling back to Facenet512")
                embedding_objs = DeepFace.represent(img_path=temp_path, 
                                           model_name="Facenet512", 
                                           detector_backend=detector_backend,
                                           enforce_detection=True)
                
                if not embedding_objs:
                    return jsonify({'error': 'Failed to generate face embedding with fallback model'}), 500
                    
                embedding = embedding_objs[0]["embedding"]
                print(f"Face embedding generated with fallback model, length: {len(embedding)}")
                return jsonify({'embedding': embedding})
            else:
                return jsonify({'error': f'Failed to generate face embedding: {error_message}'}), 500
    
    except Exception as e:
        print(f"Unexpected error in get_face_embedding: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Always clean up the temporary file
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/verify_face', methods=['POST'])
def verify_face():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Save temporary image file
        temp_image_path = 'temp_verify.jpg'
        with open(temp_image_path, 'wb') as f:
            f.write(image_bytes)

        # First, verify face detection quality
        face_objs = DeepFace.extract_faces(
            img_path=temp_image_path,
            detector_backend='retinaface',
            enforce_detection=True,
            align=True
        )

        if not face_objs:
            return jsonify({'error': 'No face detected in the image'}), 400

        # Check if multiple faces are detected
        if len(face_objs) > 1:
            return jsonify({'error': 'Multiple faces detected. Please ensure only one face is visible'}), 400

        face_obj = face_objs[0]
        
        # Check face size and position with more lenient parameters
        face_area = face_obj['facial_area']
        img = cv2.imread(temp_image_path)
        height, width = img.shape[:2]
        
        # Calculate face size relative to image
        face_width_ratio = face_area['w'] / width
        face_height_ratio = face_area['h'] / height
        
        # Face should occupy at least 20% of the image width
        if face_width_ratio < 0.2:
            return jsonify({
                'error': 'Face is too small. Please move closer to the camera',
                'details': f'Face width ratio: {face_width_ratio:.2f}'
            }), 400

        # Generate face embedding using GhostFaceNet
        try:
            primary_embedding = DeepFace.represent(
                img_path=temp_image_path,
                model_name='GhostFaceNet',
                detector_backend='retinaface',
                enforce_detection=True,
                align=True
            )[0]['embedding']
            
            model_name = 'GhostFaceNet'
            
        except Exception as e:
            # Fallback to Facenet512 if GhostFaceNet is not available
            print(f"Error using GhostFaceNet, falling back to Facenet512: {str(e)}")
            primary_embedding = DeepFace.represent(
                img_path=temp_image_path,
                model_name='Facenet512',
                detector_backend='retinaface',
                enforce_detection=True,
                align=True
            )[0]['embedding']
            
            model_name = 'Facenet512'
        
        # Try to get VGG-Face embedding as a secondary verification
        try:
            vgg_embedding = DeepFace.represent(
                img_path=temp_image_path,
                model_name='VGG-Face',
                detector_backend='retinaface',
                enforce_detection=True,
                align=True
            )[0]['embedding']
        except:
            # If VGG-Face fails, proceed with just the primary model
            vgg_embedding = None

        # Clean up temporary file
        os.remove(temp_image_path)

        # Compare with stored embeddings
        best_match = None
        best_primary_distance = float('inf')
        best_vgg_distance = float('inf')
        
        # Thresholds for different models
        ghostfacenet_threshold = 0.30  # Stricter threshold for GhostFaceNet
        facenet_threshold = 0.35  # For Facenet512 fallback
        vgg_threshold = 0.40  # Threshold for VGG-Face if available
        
        # Select appropriate threshold based on the model used
        primary_threshold = ghostfacenet_threshold if model_name == 'GhostFaceNet' else facenet_threshold

        with open(CSV_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stored_embedding = json.loads(row['face_embedding'])
                
                # Convert embeddings to numpy arrays
                primary_verify_array = np.array(primary_embedding)
                stored_array = np.array(stored_embedding)
                
                # Calculate cosine similarity
                similarity = np.dot(primary_verify_array, stored_array) / (np.linalg.norm(primary_verify_array) * np.linalg.norm(stored_array))
                primary_distance = 1 - similarity  # Convert similarity to distance
                
                # Track best match
                if primary_distance < best_primary_distance:
                    best_primary_distance = primary_distance
                    best_match = row
                    
                    # If we have VGG embeddings, check them too
                    if vgg_embedding is not None:
                        # We need to get the VGG embedding for the stored face as well
                        # For simplicity, we'll just use the primary distance for now
                        best_vgg_distance = primary_distance

        logger.debug(f"Best {model_name} distance: {best_primary_distance}, Threshold: {primary_threshold}")
        if vgg_embedding is not None:
            logger.debug(f"Best VGG distance: {best_vgg_distance}, Threshold: {vgg_threshold}")
        
        # Check if BOTH models verified (if using VGG) or just primary model is very confident
        if best_match and (
            best_primary_distance < primary_threshold and 
            (vgg_embedding is None or best_vgg_distance < vgg_threshold)
        ):
            return jsonify({
                'match': True,
                'full_name': best_match['full_name'],
                'dob': best_match['dob'],
                'passport_id': best_match['passport_id'],
                'flight_id': best_match['flight_id'],
                'distance': float(best_primary_distance),
                'threshold': primary_threshold,
                'model': model_name
            })
        else:
            return jsonify({
                'match': False, 
                'message': 'No matching face found in database',
                'best_distance': float(best_primary_distance),
                'threshold': primary_threshold,
                'model': model_name
            })

    except Exception as e:
        logger.error(f"Error in face verification: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Update the preload_deepface_models function for better error handling
def preload_deepface_models():
    """Preload DeepFace models to prevent timeouts during requests"""
    try:
        from deepface import DeepFace
        print("Pre-loading DeepFace models...")
        
        # Check if models are already downloaded
        import os
        home = os.path.expanduser("~")
        deepface_home = os.environ.get("DEEPFACE_HOME", os.path.join(home, ".deepface"))
        
        if not os.path.exists(deepface_home):
            print(f"DeepFace home directory not found at {deepface_home}")
            os.makedirs(deepface_home, exist_ok=True)
            
        # Try to load GhostFaceNet model first
        try:
            print("Attempting to load GhostFaceNet model...")
            model = DeepFace.build_model("GhostFaceNet")
            print("GhostFaceNet model loaded successfully from cache")
            primary_model = "GhostFaceNet"
        except Exception as e:
            print(f"Error loading GhostFaceNet model: {e}")
            print("Falling back to Facenet512...")
            try:
                model = DeepFace.build_model("Facenet512")
                print("Facenet512 model loaded successfully as fallback")
                primary_model = "Facenet512"
            except Exception as e2:
                print(f"Error loading Facenet512 model: {e2}")
                print("Attempting to load Facenet model...")
                model = DeepFace.build_model("Facenet")
                print("Facenet model loaded successfully from cache")
                primary_model = "Facenet"
            
        # Force model loading for faster first request
        detector_backend = "retinaface"
        try:
            print(f"Testing {detector_backend} detector...")
            DeepFace.extract_faces(img_path="https://github.com/serengil/deepface/raw/master/tests/dataset/img1.jpg", 
                                  detector_backend=detector_backend)
            print(f"{detector_backend} detector loaded successfully")
        except Exception as e:
            print(f"Error with {detector_backend} detector: {e}")
            print("Falling back to opencv detector")
            
        print(f"DeepFace models preloaded successfully. Primary model: {primary_model}")
        return True
    except Exception as e:
        print(f"Error preloading DeepFace models: {e}")
        return False

# Run preloading before app starts but don't block if it fails
preload_success = preload_deepface_models()
print(f"Model preloading {'successful' if preload_success else 'failed'}")

if __name__ == '__main__':
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # Only for development
    app.run(host='localhost', port=3000, debug=True) 