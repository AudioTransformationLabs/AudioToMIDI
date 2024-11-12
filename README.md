# AudioToMIDI

Model Application that uses ___ model to convert audio files (wav, mp3) into MIDI files.

Currently we only plan to support piano audio files.

# Steps Required

- Choose dataset(s) to use

- Choose models to test

- Write jupyter notebooks that run the model, produce outputs as well as evaluate the model.

## Run the model
- Use https://github.com/mir-dataset-loaders/mirdata/ to load dataset
- Use pretrained models - with their proprietary api
- Design and train a model with whichever kind of pretrained model performs best?

## Produce outputs
- From above - save outputs in output directory

## Evaluate model
- Evaluate with test set to get model metrics.


# Upcoming Tasks

- Implement Model Architecture
- Confirm Output and MIDI conversion to compatible formats
- Write training loop
- Write functions for evaluating the model (accuracy, F1 score, etc.)
- K-Fold Cross Validation (fine tune hyperparams)
- Evaluate on test set (and maybe other benchmark datasets to compare with baseline?)
- Write report

