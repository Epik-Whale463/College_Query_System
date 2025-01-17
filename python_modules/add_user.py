from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.email import EmailTools
import string
import random

# Load environment variables
load_dotenv()

# Configuration
MONGO_URI = os.environ.get('MONGO_URI')
APP_PASSWORD_GMAIL = os.getenv("APP_PASSWORD_GMAIL")
SENDER_EMAIL = "gcloudsignup@gmail.com"
SENDER_NAME = "System Admin"

def connect_to_db():
    try:
        client = MongoClient(MONGO_URI)
        db = client['csm_agent']
        return db
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return None

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
        return False, "Password must be at least 8 characters long"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    return True, "Password is valid"

def generate_password():
    """Generate a random password that meets the requirements."""
    while True:
        # Ensure at least one uppercase, one digit, and 10 characters
        chars = string.ascii_letters + string.digits
        password = ''.join(random.choice(chars) for _ in range(10))
        if (any(c.isupper() for c in password) and 
            any(c.isdigit() for c in password)):
            return password

def save_to_file(email, password):
    """Save credentials to user.txt file."""
    with open('users.txt', 'a') as f:
        f.write(f"Email: {email}, Password: {password}\n")

def notify_admin(user_email, user_password, notification_email):
    """Send credentials to specified email."""
    try:
        agent = Agent(
            model=Gemini(id="gemini-1.5-flash"),
            tools=[
                EmailTools(
                    receiver_email=notification_email,
                    sender_email=SENDER_EMAIL,
                    sender_name=SENDER_NAME,
                    sender_passkey=APP_PASSWORD_GMAIL,
                )
            ]
        )
        
        subject_and_body = f"""
        Subject: New User Account Created
        Body: New user account has been created with the following credentials:
        
        User Email: {user_email}
        User Password: {user_password}
        
        This information has been saved to the database and users.txt file.
        """
        
        agent.print_response(f"Send email {subject_and_body} to {notification_email}")
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

def add_user(email, notification_email=None):
    db = connect_to_db()
    if db is None:
        return False, "Database connection failed", None
    
    users_collection = db['users']
    
    # Validate email
    if not is_valid_email(email):
        return False, "Invalid email format. Must be a VVIT email address", None
    
    # Check if user already exists
    if users_collection.find_one({"email": email.lower()}):
        return False, "User already exists", None
    
    # Generate password
    password = generate_password()
    
    try:
        # Create new user
        new_user = {
            'email': email.lower(),
            'password': generate_password_hash(password),
            'created_at': datetime.now(timezone.utc)
        }
        
        users_collection.insert_one(new_user)
        
        # Save to file
        save_to_file(email, password)
        
        # Send email notification if email is provided
        if notification_email:
            if not notify_admin(email, password, notification_email):
                return True, "User created but email notification failed", password
        
        return True, "User created successfully", password
    except Exception as e:
        return False, f"Error creating user: {str(e)}", None

def main():
    print("=== CSM Agent User Creation Tool ===")
    user_email = input("Enter new user's email: ").strip()
    notification_email = input("Enter the email where credentials should be sent: ").strip()
    
    success, message, password = add_user(user_email, notification_email)
    print("\nResult:", message)
    
    if success:
        print(f"\nGenerated password: {password}")
        print("Credentials have been:")
        print("1. Saved to the database")
        print("2. Saved to users.txt")
        if notification_email:
            print(f"3. Sent to {notification_email}")
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()