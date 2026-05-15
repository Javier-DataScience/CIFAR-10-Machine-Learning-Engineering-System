# ------------------------------
# BASE IMAGE
# ------------------------------
FROM python:3.10-slim

# ------------------------------
# WORKDIR
# ------------------------------
WORKDIR /app

# ------------------------------
# SYSTEM DEPENDENCIES
# ------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# COPY PROJECT FILES
# ------------------------------
COPY . /app

# ------------------------------
# INSTALL PYTHON DEPENDENCIES
# ------------------------------
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ------------------------------
# EXPOSE FASTAPI PORT
# ------------------------------
EXPOSE 8000

# ------------------------------
# START COMMAND
# ------------------------------
CMD ["uvicorn", "fastapi.app:app", "--host", "0.0.0.0", "--port", "8000"]