import jams
import pretty_midi
import numpy as np
import os
import pandas as pd
from shutil import copy
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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


def process_guitarset(data_path, output_path):
    audio_files = sorted(
        [f for f in os.listdir(f"{data_path}/audio") if f.endswith(".wav")]
    )
    midi_files = sorted(
        [f for f in os.listdir(f"{data_path}/midi") if f.endswith(".mid")]
    )

    data = {"audio_filename": audio_files, "midi_filename": midi_files}
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["audio_filename", "midi_filename"])
        for audio_file, midi_file in zip(audio_files, midi_files):
            writer.writerow([audio_file, midi_file])


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
        copy(audio_path, output_path)
        output_path = os.path.join(output, "midi", midi_path.split("/")[-1])
        copy(midi_path, output_path)


# Create train test split by calling this function on required subsets
# Create 80-20 splits
def create_train_test_split(input_path: str, train_path: str, test_path: str) -> None:
    input_audio_path = os.path.join(input_path, "audio/")
    input_midi_path = os.path.join(input_path, "midi/")

    train_audio_path = os.path.join(train_path, "audio")
    train_midi_path = os.path.join(train_path, "midi")

    test_audio_path = os.path.join(test_path, "audio")
    test_midi_path = os.path.join(test_path, "midi")

    # input_files = zip(
    #     sorted(os.listdir(input_audio_path)), sorted(os.listdir(input_midi_path))
    # )
    input_files = [
        (input_audio, input_midi)
        for input_audio, input_midi in zip(
            sorted(os.listdir(input_audio_path)), sorted(os.listdir(input_midi_path))
        )
    ]

    train_files, test_files = train_test_split(input_files, 0.2, 0.8, shuffle=True)
    for train_audio_file, train_midi_file in train_files:
        copy(
            os.path.join(input_audio_path, train_audio_file),
            os.path.join(train_audio_path, train_audio_file),
        )
        copy(
            os.path.join(input_midi_path, train_midi_file),
            os.path.join(train_midi_path, train_midi_file),
        )

    for test_audio_file, test_midi_file in test_files:

        copy(
            os.path.join(input_audio_path, test_audio_file),
            os.path.join(test_audio_path, test_audio_file),
        )
        copy(
            os.path.join(input_midi_path, test_midi_file),
            os.path.join(test_midi_path, test_midi_file),
        )
