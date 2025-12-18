FROM python:3.11-slim

WORKDIR /app

# pastikan pip terbaru (lebih stabil)
RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY traffic_generator.py .

CMD ["python", "traffic_generator.py"]
