# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# COPY ./requirements.txt /app/requirements.txt

# RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
FROM python:3.8
 
RUN apt-get update && apt-get install -y python3-opencv 
RUN pip install --upgrade pip
RUN pip install --upgrade opencv-contrib-python 
COPY requirements.txt ./

# Install production dependencies.
RUN set -ex; \
    pip install -r requirements.txt; \
    pip install gunicorn

COPY ./app /app  
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV APP_HOME /app  

CMD exec gunicorn --bind :$PORT --workers 1 --timeout 8000 --worker-class uvicorn.workers.UvicornWorker  --threads 8 app.bot_api:app