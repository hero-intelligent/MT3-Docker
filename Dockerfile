FROM tensorflow/tensorflow:latest-gpu
# WORKDIR /

# preinstall necessary packages and dependencies
RUN apt update
RUN apt install -y git wget curl
RUN apt install -y libfluidsynth2 build-essential libasound2-dev libjack-dev
RUN git clone https://github.com/hero-intelligent/MT3-Docker.git app
WORKDIR /app
RUN pip install -r requirements.txt

# prepare environments
RUN pip install gradio gsutil
RUN git clone --branch=main https://github.com/google-research/t5x
RUN mv t5x t5x_tmp; mv t5x_tmp/* .; rm -r t5x_tmp
RUN sed -i 's:jax\[tpu\]:jax:' setup.py
RUN python3 -m pip install -e .
RUN python3 -m pip install --upgrade pip

# install mt3
RUN git clone --branch=main https://github.com/magenta/mt3
RUN mv mt3 mt3_tmp; mv mt3_tmp/* .; rm -r mt3_tmp
RUN python3 -m pip install -e .
# RUN pip install tensorflow_cpu

# copy checkpoints
RUN gsutil -q -m cp -r gs://mt3/checkpoints .

# copy soundfont (originally from https://sites.google.com/site/soundfonts4u)
RUN gsutil -q -m cp gs://magentadata/soundfonts/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2 .

RUN mkdir -p /home/user && ln -s /app /home/user/app

EXPOSE 7860

CMD [ "python", "app.py" ]
