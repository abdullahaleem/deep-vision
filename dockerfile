FROM abdullahaleem/base:pytorchgpu

# Installing additional packages
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python
RUN pip install wandb
RUN conda install -c conda-forge matplotlib

# Making working directories in the container
RUN mkdir -p /home/${maintainer}
RUN mkdir -p /home/${maintainer}/src
RUN mkdir -p /home/${maintainer}/data

# Setting the working directory in the container
WORKDIR /home/${maintainer}/src

# Copying files into the container. Useful for deployment.
# COPY ./src /home/${maintainer}/src

# Starting the terminal
#CMD ["/bin/bash"]

# Running main.py
CMD ["python", "main.py"]
