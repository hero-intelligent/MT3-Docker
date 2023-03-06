FROM tensorflow/tensorflow:latest-gpu

WORKDIR /home/user/app

# prepare environments
RUN apt update && apt install -y git libfluidsynth2 build-essential libasound2-dev libjack-dev
RUN pip install gradio gsutil
RUN git clone --branch=main https://github.com/google-research/t5x; \
    mv t5x t5x_tmp; mv t5x_tmp/* .; rm -r t5x_tmp
RUN sed -i 's:jax\[tpu\]:jax:' setup.py
RUN python3 -m pip install -e .
RUN python3 -m pip install --upgrade pip

# install mt3
RUN git clone --branch=main https://github.com/magenta/mt3; \
    mv mt3 mt3_tmp; mv mt3_tmp/* .; rm -r mt3_tmp
RUN python3 -m pip install -e .
# RUN pip install tensorflow_cpu

# copy checkpoints
RUN gsutil -q -m cp -r gs://mt3/checkpoints .

# copy soundfont (originally from https://sites.google.com/site/soundfonts4u)
RUN gsutil -q -m cp gs://magentadata/soundfonts/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2 .

COPY . .
RUN pip install -r requirements.txt

EXPOSE 7860

CMD [ "python", "app.py" ]
