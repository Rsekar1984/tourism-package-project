# Minimal Python 3.9 base image
FROM python:3.9

# Set container working directory
WORKDIR /app

# Copy all local files into the container
COPY . .

# Install all Python dependencies
RUN pip3 install -r requirements.txt

# Create a non-root user for security best practice
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory to user home
WORKDIR $HOME/app

# Copy app files with correct ownership
COPY --chown=user . $HOME/app

# Start Streamlit on port 7860 (HF Spaces default Docker port)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
