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

ENV PORT 3000

CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 8000 --worker-class uvicorn.workers.UvicornWorker  --threads 8 app.bot_api:app