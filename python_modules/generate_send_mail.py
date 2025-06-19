from phi.agent import Agent, RunResponse
from phi.model.google import Gemini
from phi.tools.email import EmailTools
from dotenv import load_dotenv
import google.generativeai as genai
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = "AIzaSyBWiOOTwv1xF7N1jt2U8GP74aPTN6jQ4ZY"
PHI_API_KEY = "phi-54SCROX5jEW0_oCzGD002hIKyHKdnuBYYaA9Xhh4NB0"
APP_PASSWORD_GMAIL = "hvjj ulxd gmvw zvlu"
EMAIL_SENDER_ADDRESS = "gcloudsignup@gmail.com"

if not all([APP_PASSWORD_GMAIL, EMAIL_SENDER_ADDRESS, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables for email functionality")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Create the model with configuration
generation_config = {
    "temperature": 0.7,  # Adjusted for more consistent output
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the model
try:
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b",
        generation_config=generation_config,
    )
    chat_session = model.start_chat(history=[])
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini model: {str(e)}")
    raise

def generate_email_content(prompt):
    """Generate email content using Gemini model"""
    try:
        # Enhance the prompt for better email generation
        enhanced_prompt = f"""
        Generate a professional email with the following requirements:
        - Purpose: {prompt}
        - Include appropriate subject line
        - Keep it concise and professional
        - Include proper greeting and closing
        - Format with clear paragraphs
        """
        
        response = chat_session.send_message(enhanced_prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating email content: {str(e)}")
        raise

def send_email_complete(to_generate, to_email):
    """
    Generate and send email using the provided content and recipient
    Returns a dictionary with status and details
    """
    try:
        # Generate email content
        logger.info(f"Generating email content for prompt: {to_generate}")
        email_content = generate_email_content(to_generate)
        
        # Setup sender details
        sender_name = "CSM Query System"
        sender_email = EMAIL_SENDER_ADDRESS
        
        # Create and configure email sender agent
        email_sender_agent = Agent(
            model=Gemini(id="gemini-1.5-flash"),
            tools=[
                EmailTools(
                    receiver_email=to_email,
                    sender_email=sender_email,
                    sender_name=sender_name,
                    sender_passkey=APP_PASSWORD_GMAIL,
                )
            ]
        )
        
        # Send the email
        logger.info(f"Sending email to: {to_email}")
        response = email_sender_agent.run(f"Send the following email to {to_email}:\n\n{email_content}")
        
        # Log success and return details
        logger.info("Email sent successfully")
        return {
            "status": "success",
            "message": "Email sent successfully",
            "details": {
                "recipient": to_email,
                "content_preview": email_content[:100] + "...",
                "full_content": email_content
            }
        }
        
    except Exception as e:
        logger.error(f"Error in send_email_complete: {str(e)}")
        raise RuntimeError(f"Failed to send email: {str(e)}")

# Optional: Add email validation function
def validate_email(email: str) -> bool:
    """Basic email validation"""
    return '@' in email and '.' in email.split('@')[1]