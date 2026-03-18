# TEST : 1) docker build --tag=$GAR_IMAGE:dev .
#        2) docker run -it -e PORT=8000 -p 8000:8000 $GAR_IMAGE:dev bash
#        3) docker run -it -e PORT=8000 -p 8000:8000 $GAR_IMAGE:dev

# TODO: select a base image
# Tip: start with a full base image, and then see if you can optimize with
#      a slim or tensorflow base

#      Standard version
FROM python:3.10.6-slim

#      Slim version
# FROM python:3.10.6-slim

#      Tensorflow version (attention: won't run on Apple Silicon)
# FROM tensorflow/tensorflow:2.16.1

# Install requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy our code
COPY cf_copilot cf_copilot
COPY setup.py setup.py

# Make directories that we need, but that are not included in the COPY
RUN mkdir /raw_data
# RUN mkdir /models # we dont need this as it lives inside packagename directory

# COPY credentials.json credentials.json

# TODO: to speed up, you can load your model from MLFlow or Google Cloud Storage at startup using
# RUN python -c 'replace_this_with_the_commands_you_need_to_run_to_load_the_model'

CMD ["uvicorn", "cf_copilot.api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
#test terminal
