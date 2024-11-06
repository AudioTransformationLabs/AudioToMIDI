import jams
import pretty_midi
import numpy as np
import os
import pandas as pd
from shutil import move


def jams_to_midi(jams_path, q=1):
    # q = 1: with pitch bend. q = 0: without pitch bend.
    jam = jams.load(jams_path)
    midi = pretty_midi.PrettyMIDI()
    annos = jam.search(namespace="note_midi")
    if len(annos) == 0:
        annos = jam.search(namespace="pitch_midi")
    for anno in annos:
        midi_ch = pretty_midi.Instrument(program=25)
        for note in anno:
            pitch = int(round(note.value))
            bend_amount = int(round((note.value - pitch) * 4096))
            st = note.time
            dur = note.duration
            n = pretty_midi.Note(
                velocity=100 + np.random.choice(range(-5, 5)),
                pitch=pitch,
                start=st,
                end=st + dur,
            )
            pb = pretty_midi.PitchBend(pitch=bend_amount * q, time=st)
            midi_ch.notes.append(n)
            midi_ch.pitch_bends.append(pb)
        if len(midi_ch.notes) != 0:
            midi.instruments.append(midi_ch)
    return midi


def create_dataset(csv_path, data_path, output, audio_col, midi_col):
    df = pd.read_csv(csv_path)
    df = df.filter([audio_col, midi_col])
    samples = df.sample(n=250)

    if not os.path.exists(output):
        os.makedirs(output)
        os.makedirs(os.path.join(output, "audio"))
        os.makedirs(os.path.join(output, "midi"))

    for _, row in samples.iterrows():
        audio_path = os.path.join(data_path, row[audio_col])
        midi_path = os.path.join(data_path, row[midi_col])
        output_path = os.path.join(output, "audio", audio_path.split("/")[-1])
        move(audio_path, output_path)
        output_path = os.path.join(output, "midi", midi_path.split("/")[-1])
        move(midi_path, output_path)
