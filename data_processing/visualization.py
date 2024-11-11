import pretty_midi
import matplotlib.pyplot as plt

def plot_midi_annotation(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    plt.figure(figsize=(15, 6))
    
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start_time = note.start
            end_time = note.end
            pitch = note.pitch
            
            plt.plot([start_time, end_time], [pitch, pitch], label=f'Note {pitch}', color='b', lw=2)

    plt.xlabel('Time (seconds)')
    plt.ylabel('MIDI Pitch')
    plt.title('MIDI Notes Visualization')
    plt.show()