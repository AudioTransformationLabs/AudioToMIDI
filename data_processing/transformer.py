import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC
from torch.nn.functional import pad
from pretty_midi import PrettyMIDI
from .constants import SAMPLE_RATE, N_FFT, N_MFCC, N_MELS, HOP_LENGTH


class Transformer:
    @staticmethod
    def mfcc_transform(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_ftt=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    ):
        return MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": n_ftt, "hop_length": hop_length, "n_mels": n_mels},
        )

    @staticmethod
    def mel_spec_transform(
        sample_rate=SAMPLE_RATE, n_ftt=N_FFT, n_mels=N_MELS, hop_length=HOP_LENGTH
    ):
        return MelSpectrogram(sample_rate, n_ftt, n_mels, hop_length)

    @staticmethod
    def transform_audio(audio_path, transform):
        waveform, sr = torchaudio.load(audio_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0).unsqueeze(dim=0)

        transformed_audio = transform(waveform)
        return transformed_audio

    @staticmethod
    def transform_midi(midi_path, num_time_frames, fs=100):
        midi = PrettyMIDI(midi_path)
        piano_roll = torch.zeros(128, num_time_frames)

        for instrument in midi.instruments:
            for note in instrument.notes:
                start_frame = int(note.start * fs)
                end_frame = int(note.end * fs)
                if end_frame > num_time_frames:
                    end_frame = num_time_frames
                piano_roll[note.pitch, start_frame:end_frame] = 1

        return piano_roll

    @staticmethod
    def split_into_chunks(tensor, chunk_length, hop_length):
        num_time_frames = tensor.shape[-1]
        chunks = []
        for start in range(0, num_time_frames - chunk_length + 1, hop_length):
            end = start + chunk_length
            chunks.append(tensor[..., start:end])

        if len(chunks) == 0:
            padding = chunk_length - tensor.shape[-1]
            if padding > 0:
                tensor = pad(tensor, (0, padding))
            return tensor[..., :chunk_length].unsqueeze(0)

        return torch.stack(chunks)

    @staticmethod
    def split_audio_midi_pair(
        audio_path, midi_path, transform, chunk_length, hop_length
    ):
        spectrogram = Transformer.transform_audio(audio_path, transform)
        num_time_frames = spectrogram.shape[-1]

        piano_roll = Transformer.transform_midi(midi_path, num_time_frames)

        audio_chunks = Transformer.split_into_chunks(
            spectrogram, chunk_length, hop_length
        )
        midi_chunks = Transformer.split_into_chunks(
            piano_roll, chunk_length, hop_length
        )

        return audio_chunks, midi_chunks

    @staticmethod
    def remove_short_fragments(predicted_midi, min_length=10):
        notes, time_frames = predicted_midi.shape
        for note in range(notes):
            segment_start = None
            for time_frame in range(time_frames):
                if segment_start is not None:
                    if predicted_midi[note][time_frame] == 0:
                        length = time_frame - segment_start
                        if length < min_length:
                            predicted_midi[note, segment_start:time_frame] = 0
                        segment_start = None
                else:
                    segment_start = time_frame if predicted_midi[note][time_frame] == 1 else segment_start

        return predicted_midi
    
    @staticmethod
    def fill_segment_gaps(predicted_midi, min_length=10):
        notes, time_frames = predicted_midi.shape
        for note in range(notes):
            prev_segment_end = 0
            for time_frame in range(time_frames):
                if prev_segment_end is not None:
                    if predicted_midi[note][time_frame] == 1:
                        length = time_frame - prev_segment_end
                        if length < min_length:
                            predicted_midi[note][prev_segment_end:time_frame] = 1
                        prev_segment_end = None
                else:
                    prev_segment_end = time_frame - 1 if predicted_midi[note][time_frame] == 0 else None

        return predicted_midi