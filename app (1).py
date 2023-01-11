BASE_DIR = "gs://download.magenta.tensorflow.org/models/music_vae/colab2"

from google.colab import files
import magenta.music as mm
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import numpy as np
import os
import tensorflow.compat.v1 as tf

import streamlit as st
from pyngrok import ngrok
import pandas as pd

from midi2audio import FluidSynth


def interpolate(model, start_seq, end_seq, num_steps, max_length=32,
                assert_same_length=True, temperature=0.5,
                individual_duration=4.0):

  """Interpolates between a start and end sequence."""
  note_sequences = model.interpolate(start_seq, end_seq,
                                       num_steps=num_steps,
                                       length=max_length,
                                       temperature=temperature,
                                       assert_same_length=assert_same_length)

  print('Start -> End Interpolation')
  interp_seq = mm.sequences_lib.concatenate_sequences(note_sequences, [individual_duration] * len(note_sequences))
  
  extracted_mid = mm.sequence_proto_to_midi_file(interp_seq, 'output1.mid')
  print(extracted_mid)
  fs = FluidSynth(sound_font='font.sf2')
  fs.midi_to_audio('output1.mid', 'output.wav')
  audio_file = open('output.wav', 'rb')
  audio_bytes = audio_file.read()
  print(audio_bytes)
  st.audio(audio_bytes, format="audio/wav")


drums_models = {}
# One-hot encoded.
drums_config = configs.CONFIG_MAP['cat-drums_2bar_small']
drums_models['drums_2bar_oh_lokl'] = TrainedModel(drums_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/checkpoints/drums_2bar_small.lokl.ckpt')
drums_models['drums_2bar_oh_hikl'] = TrainedModel(drums_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/checkpoints/drums_2bar_small.hikl.ckpt')

# Multi-label NADE.
drums_nade_reduced_config = configs.CONFIG_MAP['nade-drums_2bar_reduced']
drums_models['drums_2bar_nade_reduced'] = TrainedModel(drums_nade_reduced_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/checkpoints/drums_2bar_nade.reduced.ckpt')
drums_nade_full_config = configs.CONFIG_MAP['nade-drums_2bar_full']
drums_models['drums_2bar_nade_full'] = TrainedModel(drums_nade_full_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/checkpoints/drums_2bar_nade.full.ckpt')



# ---------------------------------------------main----------------------------------------------



page_manager = 0
count1 = 0
count2 = 0

st.title("Input midi 1")
uploaded_file1 = st.file_uploader("Choose your file", type='mid', key="input_1")
bytes_data1 = []

st.title("Input midi 2")
uploaded_file2 = st.file_uploader("Choose your file", type='mid', key="input_2")
bytes_data2 = []


if (uploaded_file1!=None) and (uploaded_file2!=None):
    bytes_data1.append(uploaded_file1.getvalue())
    drums_input_seqs1 = [mm.midi_to_sequence_proto(m) for m in bytes_data1]

    bytes_data2.append(uploaded_file2.getvalue())
    drums_input_seqs2 = [mm.midi_to_sequence_proto(m) for m in bytes_data2]
    

    extracted_beats1 = []
    for ns in drums_input_seqs1:
       extracted_beats1.extend(drums_nade_full_config.data_converter.from_tensors(drums_nade_full_config.data_converter.to_tensors(ns)[1]))

    extracted_beats2 = []
    for ns in drums_input_seqs2:
       extracted_beats2.extend(drums_nade_full_config.data_converter.from_tensors(drums_nade_full_config.data_converter.to_tensors(ns)[1]))
    
    st.text("input midi 1")
    for i, ns in enumerate(extracted_beats1):
        extracted_mid = mm.sequence_proto_to_midi_file(extracted_beats1[i], 'output1.mid')
        fs = FluidSynth(sound_font='font.sf2')
        fs.midi_to_audio('output1.mid', 'output1.wav')
        
        audio_file1 = open('output1.wav', 'rb')
        audio_bytes1 = audio_file1.read()
        st.write(i)
        count1 += 1
        st.audio(audio_bytes1, format="audio/wav")
    
    st.text("input midi 2")
    for i, ns in enumerate(extracted_beats2):
        extracted_mid2 = mm.sequence_proto_to_midi_file(extracted_beats2[i], 'output2.mid')
        fs = FluidSynth(sound_font='font.sf2')
        fs.midi_to_audio('output2.mid', 'output2.wav')
        
        audio_file2 = open('output2.wav', 'rb')
        audio_bytes2 = audio_file2.read()
        st.write(i)
        count2 += 1
        st.audio(audio_bytes2, format="audio/wav")
    
    page_manager += 1
    

st.title("Set Up")
if page_manager == 1:
    
    st.text('')
    #drums_interp_model = st.selectbox('Choose Model', ("'drums_2bar_oh_lokl'", "'drums_2bar_oh_hikl'", "'drums_2bar_nade_reduced'", "'drums_2bar_nade_full'"), key=1)
    drums_interp_model = st.selectbox('Choose Model', ("drums_2bar_oh_lokl", "drums_2bar_oh_hikl", "drums_2bar_nade_reduced", "drums_2bar_nade_full"), key=1)
    st.text('')
    temperature = st.slider('Temperature', min_value=0.1, max_value=1.5, step=0.1, key=1)
    st.text('')
    num_steps = st.number_input('Number step', min_value=1, max_value=None, step=1)


    st.text('')
    choose_start_beat = st.selectbox('Choose start beat', ('extracted_beats1', 'extracted_beats2'), key=2)
    start_num = st.slider('Choose start beat number', min_value=0, max_value=count1, key=3)
    
    if choose_start_beat=='extracted_beats1':
        start_beat = extracted_beats1[start_num]
    elif choose_start_beat=='extracted_beats2':
        start_beat = extracted_beats2[start_num]


    st.text('')
    choose_end_beat = st.selectbox('Choose end beat', ('extracted_beats1', 'extracted_beats2'), key=3)
    end_num = st.slider('Choose end beat number', min_value=0, max_value=count2, key=4)

    if choose_end_beat=='extracted_beats1':
        end_beat = extracted_beats1[end_num]
    elif choose_start_beat=='extracted_beats2':
        end_beat = extracted_beats2[end_num]


if st.button('Generate Interpolation'):

    drums_interp = interpolate(drums_models[drums_interp_model], start_beat, end_beat, num_steps=num_steps, temperature=temperature)
