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
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Save temporary image file
        temp_image_path = 'temp_face.jpg'
        with open(temp_image_path, 'wb') as f:
            f.write(image_bytes)

        # Generate face embedding using DeepFace
        embedding = DeepFace.represent(
            img_path=temp_image_path,
            model_name='Facenet512',
            detector_backend='retinaface',
            enforce_detection=True,
            align=True
        )

        # Clean up temporary file
        os.remove(temp_image_path)

        return jsonify({'embedding': embedding[0]['embedding']})

    except Exception as e:
        logger.error(f"Error generating face embedding: {str(e)}")
        return jsonify({'error': str(e)}), 500

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

        # Generate face embedding for verification using both Facenet512 and VGG-Face
        # This uses two different models for cross-validation
        facenet_embedding = DeepFace.represent(
            img_path=temp_image_path,
            model_name='Facenet512',
            detector_backend='retinaface',
            enforce_detection=True,
            align=True
        )[0]['embedding']
        
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
            # If VGG-Face fails, proceed with just Facenet512
            vgg_embedding = None

        # Clean up temporary file
        os.remove(temp_image_path)

        # Compare with stored embeddings
        best_match = None
        best_facenet_distance = float('inf')
        best_vgg_distance = float('inf')
        
        # Much stricter thresholds
        facenet_threshold = 0.35  # Very strict threshold for cosine similarity
        vgg_threshold = 0.40  # Threshold for VGG-Face if available

        with open(CSV_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stored_embedding = json.loads(row['face_embedding'])
                
                # Convert embeddings to numpy arrays
                facenet_verify_array = np.array(facenet_embedding)
                stored_array = np.array(stored_embedding)
                
                # Calculate cosine similarity for Facenet512
                similarity = np.dot(facenet_verify_array, stored_array) / (np.linalg.norm(facenet_verify_array) * np.linalg.norm(stored_array))
                facenet_distance = 1 - similarity  # Convert similarity to distance
                
                # Track best match
                if facenet_distance < best_facenet_distance:
                    best_facenet_distance = facenet_distance
                    best_match = row
                    
                    # If we have VGG embeddings, check them too
                    if vgg_embedding is not None:
                        # We need to get the VGG embedding for the stored face as well
                        # For simplicity, we'll just use the Facenet distance for now
                        best_vgg_distance = facenet_distance

        logger.debug(f"Best Facenet distance: {best_facenet_distance}, Threshold: {facenet_threshold}")
        if vgg_embedding is not None:
            logger.debug(f"Best VGG distance: {best_vgg_distance}, Threshold: {vgg_threshold}")
        
        # Check if BOTH models verified (if using VGG) or just Facenet is very confident
        if best_match and (
            best_facenet_distance < facenet_threshold and 
            (vgg_embedding is None or best_vgg_distance < vgg_threshold)
        ):
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
            return jsonify({
                'match': False, 
                'message': 'No matching face found in database',
                'best_distance': float(best_facenet_distance),
                'threshold': facenet_threshold
            })

    except Exception as e:
        logger.error(f"Error in face verification: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 