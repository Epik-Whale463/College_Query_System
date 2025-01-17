from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from python_modules.rag_functions import initialize_rag, query_rag
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import logging
from datetime import datetime, timedelta, date
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.email import EmailTools
import random
import json


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Custom JSON encoder to handle datetime objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.strftime("%Y-%m-%d")
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = CustomJSONEncoder
app.secret_key = os.environ.get('SECRET_KEY', 'forloginyouneedpasswordbrother')
app.permanent_session_lifetime = timedelta(hours=2)

# MongoDB Configuration
MONGO_URI = os.environ.get('MONGO_URI')
APP_PASSWORD_GMAIL = os.getenv("APP_PASSWORD_GMAIL")
SENDER_EMAIL = os.getenv("EMAIL_SENDER_ADDRESS")

sender_email = SENDER_EMAIL
sender_name = "CSM Agent"
sender_passkey = APP_PASSWORD_GMAIL

try:
    client = MongoClient(MONGO_URI)
    db = client['csm_agent']
    users_collection = db['users']
    queries_collection = db['queries']
    otps_collection = db['otps']
    # Test the connection
    client.server_info()
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

def get_current_date():
    """Helper function to get current date without timezone information"""
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

def generate_otp():
    return ''.join(str(random.randint(0, 9)) for _ in range(6))

def send_email(receiver_email, subject, body):
    try:
        agent = Agent(
            model=Gemini(id="gemini-1.5-flash"),
            tools=[
                EmailTools(
                    receiver_email=receiver_email,
                    sender_email=sender_email,
                    sender_name=sender_name,
                    sender_passkey=sender_passkey,
                )
            ]
        )
        agent.print_response(f"Send email {subject}\n{body} to {receiver_email}")
        logger.info(f"Email sent successfully to {receiver_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        return False

def store_otp(email, otp, purpose):
    try:
        # Remove any existing OTPs for this email and purpose
        otps_collection.delete_many({
            'email': email,
            'purpose': purpose
        })
        
        current_time = datetime.utcnow()
        # Store new OTP with expiration time (10 minutes)
        otps_collection.insert_one({
            'email': email,
            'otp': otp,
            'purpose': purpose,
            'created_at': current_time,
            'expires_at': current_time + timedelta(minutes=10)
        })
        return True
    except Exception as e:
        logger.error(f"Failed to store OTP: {str(e)}")
        return False

def verify_otp(email, otp, purpose):
    try:
        otp_record = otps_collection.find_one({
            'email': email,
            'otp': otp,
            'purpose': purpose,
            'expires_at': {'$gt': datetime.utcnow()}
        })
        return otp_record is not None
    except Exception as e:
        logger.error(f"Failed to verify OTP: {str(e)}")
        return False

# Initialize RAG system
dir_path = "./data"
try:
    initialize_rag(dir_path)
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    raise

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def is_valid_email(email):
    return email.lower().endswith(("@vvit.net"))

def validate_password(password):
    """
    Validate password meets minimum requirements:
    - At least 8 characters long
    - Contains at least one number
    - Contains at least one uppercase letter
    """
    if len(password) < 8:
        return False
    if not any(c.isdigit() for c in password):
        return False
    if not any(c.isupper() for c in password):
        return False
    return True

@app.route("/")
def index():
    return redirect(url_for('login'))

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get('logged_in'):
        return redirect(url_for('home'))

    if request.method == "POST":
        email = request.form.get("email", "").lower()
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        gmail = request.form.get("gmail")

        # Validation checks
        if not email or not password or not confirm_password or not gmail:
            flash("Please fill in all fields", "error")
            return render_template("register.html")

        if not is_valid_email(email):
            flash("Please use a valid VVIT email address", "error")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match", "error")
            return render_template("register.html")

        if not validate_password(password):
            flash("Password must be at least 8 characters long and contain at least one number and one uppercase letter", "error")
            return render_template("register.html")

        # Check if user already exists
        if users_collection.find_one({"email": email}):
            flash("Email already registered", "error")
            return render_template("register.html")

        # Check if Gmail is already used for registration
        if users_collection.find_one({"gmail": gmail}):
            flash("Gmail already used for registration", "error")
            return render_template("register.html")

        # Generate and send OTP
        otp = generate_otp()
        if not store_otp(email, otp, 'registration'):
            flash("Error generating OTP. Please try again.", "error")
            return render_template("register.html")

        # Store temporary user data in session
        session['temp_user'] = {
            'email': email,
            'password': generate_password_hash(password),
            'gmail': gmail
        }

        # Send OTP email
        if not send_email(
            gmail,
            "Email Verification OTP",
            f"Your OTP for CSM Agent registration is: {otp}"
        ):
            flash("Error sending OTP email. Please try again.", "error")
            return render_template("register.html")

        return redirect(url_for('verify_registration_otp'))

    return render_template("register.html")

@app.route('/verify-registration-otp', methods=['GET', 'POST'])
def verify_registration_otp():
    if 'temp_user' not in session:
        return redirect(url_for('register'))

    if request.method == 'POST':
        entered_otp = request.form.get('otp')
        email = session['temp_user']['email']

        if verify_otp(email, entered_otp, 'registration'):
            try:
                # Save user to database
                users_collection.insert_one({
                    'email': email,
                    'password': session['temp_user']['password'],
                    'created_at': datetime.utcnow()
                })
                
                # Clear temporary data
                session.pop('temp_user', None)
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                logger.error(f"Error during registration: {str(e)}")
                flash("An error occurred during registration", "error")
                return render_template("verify_otp.html", purpose='registration')
        
        flash('Invalid or expired OTP', 'error')
    
    return render_template("verify_otp.html", purpose='registration')

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get('logged_in'):
        return redirect(url_for('home'))

    if request.method == "POST":
        email = request.form.get("email", "").lower()
        password = request.form.get("password")

        if not email or not password:
            flash("Please provide both email and password", "error")
            return render_template("login.html")

        if not is_valid_email(email):
            flash("Please use a valid VVIT email address", "error")
            return render_template("login.html")

        user = users_collection.find_one({"email": email})

        if user and check_password_hash(user['password'], password):
            session.permanent = True
            session['logged_in'] = True
            session['user_email'] = email
            session['first_login'] = True
            
            # Safely get created_at date with fallback
            created_at = user.get('created_at', datetime.utcnow())
            session['user_since'] = created_at.strftime("%B %d, %Y")
            
            flash(f'Welcome back, {email}!', 'success')
            return redirect(url_for('home'))
        else:
            flash("Invalid email or password", "error")
            return render_template("login.html")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash('Successfully logged out!', 'success')
    return redirect(url_for('login'))

@app.route("/home")
@login_required
def home():
    # Get user data for display
    user_data = {
        'email': session.get('user_email'),
        'member_since': session.get('user_since'),
        'first_login': session.pop('first_login', False)  # Remove the flag after first use
    }
    return render_template("index.html", user_data=user_data)

@app.route("/user_info")
@login_required
def user_info():
    user_email = session.get('user_email')
    user = users_collection.find_one({"email": user_email})
    
    if user:
        user_info = {
            'email': user_email,
            'member_since': user['created_at'].strftime("%B %d, %Y"),
            'query_count_today': get_today_query_count(user_email)
        }
        return jsonify(user_info)
    return jsonify({'error': 'User not found'}), 404

def get_today_query_count(email):
    current_date = get_current_date()
    query_entry = queries_collection.find_one({
        "email": email,
        "date": current_date
    })
    return query_entry['count'] if query_entry else 0

@app.route("/query", methods=["POST"])
@login_required
def user_query():
    try:
        user_email = session.get('user_email')
        current_date = get_current_date()
        
        if "charan@vvit.net" not in user_email.lower():
            query_entry = queries_collection.find_one({
                "email": user_email,
                "date": current_date
            })
            
            if query_entry:
                if query_entry['count'] >= 50:
                    logger.warning(f"Query limit exceeded for user: {user_email}")
                    return jsonify({"error": "Query limit exceeded for today."}), 429
                
                queries_collection.update_one(
                    {"email": user_email, "date": current_date},
                    {"$inc": {"count": 1}}
                )
            else:
                queries_collection.insert_one({
                    "email": user_email,
                    "date": current_date,
                    "count": 1
                })

        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
            
        query = data.get('query')
        result = query_rag(query)
        
        return jsonify({"result": result})
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": "An error occurred while processing your query"}), 500

@app.route("/query-count")
@login_required
def get_query_count():
    try:
        user_email = session.get('user_email')
        current_date = get_current_date()
        
        query_entry = queries_collection.find_one({
            "email": user_email,
            "date": current_date
        })
        count = query_entry['count'] if query_entry else 0
        
        response_data = {
            "email": user_email,
            "date": current_date.strftime("%Y-%m-%d"),
            "query_count": count,
            "queries_remaining": 50 - count
        }
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error fetching query count: {str(e)}")
        return jsonify({"error": "Failed to fetch query count"}), 500

@app.route("/healthz")
def health_check():
    try:
        # Check MongoDB connection
        client.server_info()
        return jsonify({
            "status": "healthy",
            "mongodb": "connected",
            "rag_system": "initialized"
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))  
    app.run(
        host="0.0.0.0",  
        port=port,
        debug=True
    )