# College Query System

**A Flask-based web application for handling college-related queries with RAG (Retrieval-Augmented Generation) implementation.**

## Features

- **User Authentication System**
  - Email-based registration with OTP verification
  - Secure login with password hashing
  - Session management with 2-hour lifetime

- **Query Processing**
  - RAG-based query answering system
  - Daily query limit tracking (50 queries per user)
  - Special privileges for admin users

- **Security**
  - Password validation requirements
  - Email domain restriction (@vvit.net)
  - OTP expiration after 10 minutes

- **MongoDB Integration**
  - User management
  - Query tracking
  - OTP storage and verification

- **Email Integration**
  - OTP delivery system
  - Gemini 1.5 Flash model integration
  - Automated email notifications

## Prerequisites

- Python 3.x
- MongoDB
- Gmail account for sending emails
- Required Python packages (see Installation)

## Environment Variables

Create a `.env` file with the following variables:

```
SECRET_KEY=your_secret_key
MONGO_URI=your_mongodb_connection_string
APP_PASSWORD_GMAIL=your_gmail_app_password
EMAIL_SENDER_ADDRESS=your_sender_email
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Epik-Whale463/College_Query_System.git
   cd College_Query_System
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up MongoDB**
   - Create a MongoDB database named 'csm_agent'
   - Configure the following collections:
     - users
     - queries
     - otps

4. **Initialize the RAG system**
   - Place your data files in the `./data` directory
   - System will automatically initialize on startup

## Usage

1. **Start the server**
   ```bash
   python app.py
   ```

2. **Access the application**
   - Open a web browser and navigate to `http://localhost:8080`
   - Default port is 8080, can be modified via PORT environment variable

## API Endpoints

- **Authentication**
  - `POST /register` - User registration
  - `POST /login` - User login
  - `GET /logout` - User logout

- **Query System**
  - `POST /query` - Submit a query
  - `GET /query-count` - Get daily query usage

- **User Management**
  - `GET /user_info` - Get user information
  - `GET /healthz` - System health check

## Security Features

- **Password Requirements**
  - Minimum 8 characters
  - At least one uppercase letter
  - At least one number

- **Email Verification**
  - 6-digit OTP system
  - 10-minute OTP validity
  - Domain restriction to @vvit.net

## Error Handling

- Custom error pages for 404 and 500 errors
- Comprehensive logging system
- User-friendly flash messages

## Rate Limiting

- 50 queries per user per day
- Exception for admin users (charan@vvit.net)
- Query count resets daily

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request


## Support

For support, email pvrcharan2022@gmail.com or create an issue in the repository.
