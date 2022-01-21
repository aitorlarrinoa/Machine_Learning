FROM python:3.8-slim-buster

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# copy every content from the local file to the image
COPY . /app

RUN pip install -r requirements.txt


CMD ["python" "app.py"]

