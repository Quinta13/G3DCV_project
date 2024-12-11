import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.signal import correlate

from src.model import VideoFile, AudioFile
from src.utils.io_ import FormatLogger, SilentLogger, InputSanitization as IS, IOOperations as IO, PathOperations as PO


class VideoSync:

    AUDIO_EXT = 'wav'

    def __init__(
        self, 
        video1  : VideoFile, 
        video2  : VideoFile,
        out_dir : str, 
        logger  : FormatLogger = SilentLogger()
    ):
        
        exp1 = PO.get_file_name(video1.path)
        exp2 = PO.get_file_name(video2.path)

        if exp1 != exp2:
            raise ValueError(
                f"Experiments must have the same name as they refer to the same experiment, "\
                f"got {video1.name} and {video2.name}. Please check input videos match the same experiment."
            )
        
        self._video1  : VideoFile    = video1
        self._video2  : VideoFile    = video2
        self._exp_name: str          = exp1
        self._logger  : FormatLogger = logger

        IS.check_input(out_dir)
        sync_dir = os.path.join(out_dir, 'sync')
        IO.make_dir(path=sync_dir, logger=self._logger)
        self._out_dir : str = sync_dir

    @property
    def exp_name(self) -> str: return self._exp_name

    def __str__(self) -> str:
        return f"VideoSync[{self.exp_name}]"

    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def videos(self) -> Tuple[VideoFile, VideoFile]: return self._video1, self._video2

    @property
    def out_dir(self) -> str: return self._out_dir

    def sync(self):

        # Extract audio from the videos
        audio1, audio2 = self._extract_audio()
    
        # Calculate the offset between the two audio signals
        offset = self.calculate_sync_offset(audio1=audio1, audio2=audio2)
        self._logger.info(f"Calculated offset: {offset} seconds")

        # Plot audio signals before and after computed alignement
        for off, title in zip([0., offset], ['Original', 'Aligned']):

            self.plot_aligned_audio(
                audio1=audio1,
                audio2=audio2,
                offset=off,
                title=f'{self._exp_name} {title} Audio Signals',
                save_path=os.path.join(self.out_dir, f'{title}.png'),
                logger=self._logger
            )
        


    def _extract_audio(self) -> Tuple[AudioFile, AudioFile]:

        # Compute the minimum sample rate
        min_sample_rate = min(self._video1.metadata.sample_rate, self._video2.metadata.sample_rate)

        audios: List[AudioFile] = []

        # Extract camera name
        for video in [self._video1, self._video2]:

            camera_name = PO.get_containing_folder(video.path)
            out_path = os.path.join(self._out_dir, f'{camera_name}.{self.AUDIO_EXT}')
            
            # Extract audio from the videos
            audios.append(
                VideoFile.extract_audio(
                    video=video,
                    out_path=out_path,
                    sample_rate=min_sample_rate,
                    logger=self._logger
                )
            )

        audio1, audio2 = audios
        return audio1, audio2

    @staticmethod
    def calculate_sync_offset(
            audio1: AudioFile,
            audio2: AudioFile
        ) -> float:

        # Ensure the sample rates are the same
        if audio1.rate != audio2.rate:
            raise ValueError(f"Sample rates must be the same, got {audio1.rate} and {audio2.rate}")

        # Compute the offset between the two audio signals
        correlation = correlate(audio1.data, audio2.data, mode='full')
        lag = np.argmax(correlation) - (len(audio2) - 1)

        # Compute offset in seconds
        offset_time = lag / audio1.rate

        return float(offset_time)

    @staticmethod
    def plot_aligned_audio(
        audio1    : AudioFile,
        audio2    : AudioFile,
        offset    : float = 0,
        title     : str   = '',
        save_path : str = '',
        logger    : FormatLogger = SilentLogger()
    ):
        
        # Ensure the sample rates are the same
        if audio1.rate != audio2.rate:
            raise ValueError(f"Sample rates must be the same, got {audio1.rate} and {audio2.rate}")
        
        # Convert offset from seconds to samples
        offset_samples = int(offset * audio1.rate)
    
        # Padding for signal alignment
        pad1, pad2 = (0, offset_samples), (offset_samples, 0)

        if offset < 0:
            pad1, pad2 = pad2, pad1
        
        data1 = np.pad(audio1.data, pad1, mode='constant')
        data2 = np.pad(audio2.data, pad2, mode='constant')
        
        # Truncate to the same length
        min_length = min(len(data1), len(data2))
        data1 = data1[:min_length]
        data2 = data2[:min_length]
    
        # Generate the time axis
        time = np.linspace(0, min_length / audio1.rate, num=min_length)
        
        # Plot the aligned signals
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot signals
        ax.plot(time, data2, label=audio2.name, color='orange', alpha=.6)
        ax.plot(time, data1, label=audio1.name, color='blue',   alpha=.6)
        
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True)

        if save_path:
            logger.info(f"Saving plot to {save_path}")
            plt.savefig(save_path)

        plt.tight_layout()
        plt.show()
