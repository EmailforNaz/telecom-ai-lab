# Base image with Python runtime
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI default port
EXPOSE 8000

# Start FastAPI app using uvicorn
CMD ["uvicorn", "telecom_api:app", "--host", "0.0.0.0", "--port", "8000"]