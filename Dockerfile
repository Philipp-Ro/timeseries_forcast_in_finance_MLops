# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies except pywin32
RUN sed -i '/pywin32/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app /app
COPY ./Model_DB /Model_DB

# Expose the port that your app runs on (if applicable)
# EXPOSE 8000

# Set permissions for the model_db directory
RUN chmod -R 777 /Model_DB

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]