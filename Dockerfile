# Dockerfile
#FROM python:3.9
#WORKDIR /app
#COPY . .
#RUN pip install -r requirements.txt
#CMD ["streamlit", "run", "app.py"]
# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Avoid unnecessary cache to reduce size
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (for image processing and DICOM)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
