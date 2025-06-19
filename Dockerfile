# Use a lightweight Python image
FROM python:3.10.3-slim

# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the port Flask runs on
EXPOSE 8080

# Run the Flask application
CMD ["python", "app.py"]