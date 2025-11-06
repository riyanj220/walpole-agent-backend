# 1. Base Image
# Use an official, lightweight Python image
FROM python:3.11-slim-bookworm

# 2. Set Environment Variables
# Prevents Python from writing .pyc files and buffers output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Set Work Directory
WORKDIR /app

# 4. Install Dependencies
# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# 5. Copy Your Project
# This copies ALL your code, including your 'core' app
# and, most importantly, your 'core/data/walpole' FAISS index
COPY . .

# 6. Collect Static Files
# Gathers all static files (CSS, JS) into one directory for Nginx
RUN python manage.py collectstatic --noinput

# Port 8000 is now an internal port for Gunicorn
# Nginx will talk to this port.
EXPOSE 8000

# 7. Run the Application
# START THE PRODUCTION SERVER (Gunicorn)
# !!! IMPORTANT: Replace 'your_project_name' with your actual project name (the one with wsgi.py)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "backend.wsgi:application"]