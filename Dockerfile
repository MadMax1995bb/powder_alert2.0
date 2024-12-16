FROM python:3.10-slim

WORKDIR /app

COPY requirements_prod.txt requirements.txt
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD uvicorn powderalert.api.fast:app --host 0.0.0.0 --port $PORT
