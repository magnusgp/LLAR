"""This script remixes audio files, both by mixing different class files and by mixing different files of the same class."""

import librosa

class Mixer:
    """Mixer is responsible for mixing audio files. 
       It follows the following pipeline:
       1. Load audio files from input path
       2. Mix audio files depending on the input array
       3. Save mixed audio file to output path
    """

    def __init__(self, data_path, output_path, sample_rate=16000, duration=5):
        self.data_path = data_path
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.duration = duration

    def mix(self, files):
        """Mixes the audio files in the input array.

        Args:
            files (array of strings): Array containing strings with filenames that should
            be used as an input vector. Be careful not to load a lot of files here, as it
            plots for every file.
        """
        # load the audio files
        audio = [self._load_audio(file) for file in files]
        # mix the audio files
        mixed_audio = self._mix_audio(audio)
        # save the mixed audio
        self._save_audio(mixed_audio)

    def _load_audio(self, file):
        """Loads the audio file and resamples it to the sample rate of the model.

        Args:
            file (string): Path to the audio file.

        Returns:
            array: Resampled audio file.
        """
        # load the audio file
        audio, sr = librosa.load(file, sr=self.sample_rate)
        # resample the audio file
        audio = librosa.resample(audio, sr, self.sample_rate)
        return audio

    def _mix_audio(self, audio):
        """Mixes the audio files in the input array.

        Args:
            audio (array of arrays): Array containing arrays with audio files.

        Returns:
            array: Mixed audio file.
        """
        # mix the audio files
        mixed_audio = np.sum(audio, axis=0)
        return mixed_audio

    def _save_audio(self, audio):
        """Saves the audio file to the output path.

        Args:
            audio (array): Array containing the audio file.
        """
        # save the mixed audio
        librosa.output.write_wav(self.output_path, audio, self.sample_rate)
        