import gradio as gr
from pathlib import Path

from imports_and_definitions import InferenceModel
from imports_and_definitions import upload_audio
import note_seq

def inference(audio,model):
  with open(audio, 'rb') as fd:
      contents = fd.read()
  audio = upload_audio(contents,sample_rate=16000)
  inference_model = InferenceModel('/home/user/app/checkpoints/' + model + '/', model)
  est_ns = inference_model(audio)
  note_seq.sequence_proto_to_midi_file(est_ns, './transcribed.mid')
  return './transcribed.mid'
  
title = "MT3"
description = """
Gradio demo for MT3: Multi-Task Multitrack Music Transcription. To use it, simply upload your audio file, or click one of the examples to load them. Read more at the links below.
Choose either ismir2021 for piano transcription or mt3 for multi-instrument transcription.
It will be of better quality if pure music is inputted. It is recomended to remove the voice in a song using UVR5. Check it out in the links below.
"""

article = """
<p style='text-align: center'>
    <a href='https://arxiv.org/abs/2111.03017' target='_blank'>MT3: Multi-Task Multitrack Music Transcription</a> | 
    <a href='https://github.com/magenta/mt3' target='_blank'>Github Repo</a> | 
    <a href='https://huggingface.co/spaces/oniati/mrt/tree/main' target='_blank'>Huggingface</a> | 
    <a href='https://github.com/hero-intelligent/MT3-Docker' target='_blank'>Docker Source Code</a>
</p>
<p style='text-align: center'>
    <a href='https://ultimatevocalremover.com/' target='_blank'>UVR5</a> | 
    <a href='https://github.com/Anjok07/ultimatevocalremovergui' target='_blank'>Github Repo</a>
</p>
"""

demo = gr.Interface(
    fn=inference, 
    inputs=[
      gr.inputs.Audio(type="filepath", label="Input"), 
      gr.Dropdown(choices=["mt3", "ismir2021", "ismir2022_base", "ismir2022_small"])
    ],
    [gr.outputs.File(label="Output")],
    title=title,
    description=description,
    article=article,
    allow_flagging=False,
    allow_screenshot=False,
    enable_queue=True
    )
demo.launch(server_name="0.0.0.0")
