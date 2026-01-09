FROM python:3.10-slim

WORKDIR /app

# Librerías del sistema para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar y instalar dependencias primero (mejor cache)
COPY deps/requirements.txt ./deps/
RUN pip install --no-cache-dir -r deps/requirements.txt

# Copiar el resto del proyecto
COPY . /app

ENV FLASK_APP=server.py

CMD ["flask", "run", "--host=0.0.0.0"]