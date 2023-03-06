import gradio as gr
from pathlib import Path

from imports_and_definitions import InferenceModel
from imports_and_definitions import upload_audio
import note_seq

inference_model = InferenceModel('/home/user/app/checkpoints/mt3/', 'mt3')

def inference(audio):
  with open(audio, 'rb') as fd:
      contents = fd.read()
  audio = upload_audio(contents,sample_rate=16000)
  est_ns = inference_model(audio)
  note_seq.sequence_proto_to_midi_file(est_ns, './transcribed.mid')
  return './transcribed.mid'
  
title = "MT3"
description = """
Gradio demo for MT3: Multi-Task Multitrack Music Transcription. 
To use it, simply upload your audio file, or click one of the examples to load them. 
Read more at the links below.
"""

article = """
<p style='text-align: center'>
    <a href='https://arxiv.org/abs/2111.03017' target='_blank'>MT3: Multi-Task Multitrack Music Transcription</a> | 
    <a href='https://github.com/magenta/mt3' target='_blank'>Github Repo</a>
</p>
<p style='text-align: center'>
    Pulled and adjusted from
    <a href='https://huggingface.co/spaces/oniati/mrt/tree/main' target='_blank'>This Repository</a> | 
    <a href='https://github.com/hero-intelligent/MT3-Docker' target='_blank'>Docker Source Code</a>
</p>
"""

demo = gr.Interface(
    inference, 
    gr.inputs.Audio(type="filepath", label="Input"), 
    [gr.outputs.File(label="Output")],
    title=title,
    description=description,
    article=article,
    allow_flagging=False,
    allow_screenshot=False,
    enable_queue=True
    )
demo.launch(server_name="0.0.0.0")
