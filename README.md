# MT3

Gradio demo for MT3: Multi-Task Multitrack Music Transcription. To use it, simply upload your audio file, or click one of the examples to load them. Read more at the links below.

Pulled and adjusted from: [This Huggingface Repository](https://huggingface.co/spaces/oniati/mrt)

Source codes: [MT3: Multi-Task Multitrack Music Transcription](https://arxiv.org/abs/2111.03017) | [Github Repo](https://github.com/magenta/mt3)

Another Dockerfile which may be awsome: https://github.com/jsphweid/mt3-docker

## Preparation

To run this docker container, **Ubuntu 20.04** *baremetal localhost* is recomended. You can certainly choose to run whether baremetal or on a virtual machine, whether localhost or remotely under the same LAN, or even on a remote cloud computer, but there might be some issues. the installation will be exact the same.

If you have not installed docker, please run
```Shell
sudo apt update && sudo apt install -y curl git
sudo curl -fsSL https://get.docker.com | sudo bash -s docker
```

If you have a GPU, please run
```Shell
sudo sh nvidia-container-runtime-script.sh
sudo apt-get install nvidia-container-runtime
```

## Installation and Run
```shell
git clone https://github.com/hero-intelligent/MT3-Docker.git
sudo docker build -t mt3 .
```
Then run
```Shell
sudo docker run -d --gpus=all --name=mt3 -p 7860:7860 mt3 
```

If you have **CPU only** or want to run on Windows docker, then run
```shell
sudo docker run -d --name=mt3 -p 7860:7860 mt3 
```

Finally, visit `localhost:7860` and enjoy! If you run it remotely, please change `localhost` to the remote ip instead.

# Cooperate with UVR5

The quality of transcription will be better if a pure instrumental piece is inputed. If you want to transcribe a song with vocal, it is highly recomended to remove the vocal part using the **UVR5**.

**UVR5**, or *Ultimate Vocal Remover GUI v5.5.0*, is an AI powered and self-contained vocal remover, which is supported on both Windows and GUI Linux.

Check out more details at the [Official Website](https://ultimatevocalremover.com/) and [Github Repo](https://github.com/Anjok07/ultimatevocalremovergui) and follow the instructions to have the software installed.
