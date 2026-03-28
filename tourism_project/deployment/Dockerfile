# Minimal Python 3.9 base image
FROM python:3.9

# Set container working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install all Python dependencies
RUN pip3 install -r requirements.txt

# Create non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Start Streamlit on port 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
