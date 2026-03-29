FROM python:3.10-slim

WORKDIR /app

# Install dependencies as root (required for system packages)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user (HF Spaces security requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Port 7860 is the REQUIRED port for HF Docker Spaces
EXPOSE 7860

# app.py is our file — not streamlit_app.py (the HF template default)
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false"]
