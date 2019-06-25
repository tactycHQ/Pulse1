
# Using the official tensorflow serving image from docker hub as base image
FROM tensorflow/serving

# Installing NGINX, used to rever proxy the predictions from SageMaker to TF Serving
RUN apt-get update &&  \
    apt-get install -y python-pip python-dev &&\
    pip install --upgrade pip

# Copy our model folder to the container
COPY served_models /served_models
EXPOSE 8501

# starts NGINX and TF serving pointing to our model
ENTRYPOINT ["bin/bash"]
