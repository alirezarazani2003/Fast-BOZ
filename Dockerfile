FROM python:3.10-slim

WORKDIR /app

COPY reqi.txt .
RUN pip install --no-cache-dir -r reqi.txt


COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]