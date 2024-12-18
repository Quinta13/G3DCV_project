import os
import numpy as np
from typing import List, Tuple
import ffmpeg
import matplotlib.pyplot as plt
from scipy.signal import correlate

from src.utils.io_ import (
    BaseLogger, SilentLogger, 
    IOUtils, PathUtils, InputSanitizationUtils as ISUtils,
    AudioFile, VideoFile
)
from src.utils.misc import Timer

class VideoSync:

    AUDIO_EXT     = 'wav'
    VIDEO_EXT     = 'mp4'

    def __init__(
        self, 
        video1  : VideoFile, 
        video2  : VideoFile,
        out_dir : str, 
        logger  : BaseLogger = SilentLogger(),
        verbose : bool = False
    ):
        '''
        Receive in input two video files to synchronize and an output directory to save the synchronization data.
        '''

        # Check if the videos refer to the same experiment - i.e. have the same name
        exp1 = PathUtils.get_file_name(video1.path)
        exp2 = PathUtils.get_file_name(video2.path)

        if exp1 != exp2:
            logger.handle_error(
                msg=f"Experiments must have the same name as they refer to the same experiment, "\
                    f"got {exp1} and {exp2}. Please check input videos match the same experiment.",
                exception=ValueError
            )
        
        self._video1  : VideoFile    = video1
        self._video2  : VideoFile    = video2
        self._exp_name: str          = exp1

        self._logger        : BaseLogger = logger
        self._logger_verbose: BaseLogger = logger if verbose else SilentLogger()
        self._out_dir       : str        = out_dir

    # --- MAGIC METHODS ---

    def __str__(self) -> str:
        return f"VideoSync[{self.exp_name}]"

    def __repr__(self) -> str:
        return self.__str__()


    # --- PROPERTIES ---

    @property
    def exp_name(self) -> str: return self._exp_name

    @property
    def videos(self) -> Tuple[VideoFile, VideoFile]: return self._video1, self._video2

    @property
    def out_dir(self) -> str: return self._out_dir


    # --- SYNC METHODS ---

    
    def sync(self):

        PREFIX = lambda x: f' > {x}'

        def start(msg: str): self._logger.info(msg);         self._logger.formatter = PREFIX
        def end()          : self._logger.reset_formatter(); self._logger.info('')

        timer = Timer()

        self._logger.info(f"SYNCING VIDEOS: ")
        self._logger.info(f" 1. {self._video1}")
        self._logger.info(f" 2. {self._video2}")
        self._logger.info(f"")

        # Extract audio from the videos
        start("EXTRACTING AUDIO FROM VIDEOS")
        audio1, audio2 = self._extract_audio()
        end()
    
        # Calculate the offset between the two audio signals
        start("CALCULATING AUDIO SIGNALS OFFSET")
        offset = self.calculate_sync_offset(audio1=audio1, audio2=audio2)
        self._logger.info(f"Calculated offset: {offset} seconds.")
        end()

        # Plot audio signals before and after computed alignment
        start("PLOTTING AUDIO SIGNALS")
        for off, title in zip([0., offset], ['Original', 'Aligned']):

            self.plot_aligned_audio(
                audio1=audio1,
                audio2=audio2,
                offset=off,
                title=f'{self._exp_name} {title} Audio Signals',
                save_path=os.path.join(self.out_dir, f'audio-sync_{title.lower()}.png'),
                logger=self._logger
            )
        end()
        
        # Save trimmed audios
        start("TRIMMING AUDIO SIGNALS")
        video1_out, video2_out = self.trim_video(
            video1=self._video1,
            video2=self._video2,
            sync_time=offset,
            out_path=self.out_dir,
            logger=self._logger
        )
        end()

        self._logger.info(f"SYNC COMPLETED in {timer}.")
        self._logger.info(f"")

        self._logger.info(f"SYNCED VIDEOS: ")
        self._logger.info(f" 1. {video1_out}")
        self._logger.info(f" 2. {video2_out}")
        self._logger.info(f"")

    def _extract_audio(self) -> Tuple[AudioFile, AudioFile]:
        ''' Extract audio from the two videos with the same sample rate. '''

        # Compute the minimum sample rate
        min_sample_rate = min(self._video1.metadata.sample_rate, self._video2.metadata.sample_rate)
        self._logger.info(f"Extracting audio files from videos with sample rate: {min_sample_rate}")

        # Extract audio for both the videos
        audios: List[AudioFile] = []

        for video in [self._video1, self._video2]:

            self._logger.info(f"Extracting audio from {video.name} ...")

            # Extract camera name
            camera_name = PathUtils.get_folder_name(video.path)
            out_path = os.path.join(self._out_dir, f'{camera_name}.{self.AUDIO_EXT}')
            
            # Extract audio from the video
            audios.append(
                VideoFile.extract_audio(
                    video=video,
                    out_path=out_path,
                    sample_rate=min_sample_rate,
                    logger=self._logger
                )
            )

            self._logger.info(f"")

        audio1, audio2 = audios
        return audio1, audio2
    
    @staticmethod
    def calculate_sync_offset(
            audio1: AudioFile,
            audio2: AudioFile,
            logger: BaseLogger = SilentLogger()
        ) -> float:
        ''' Computes the offset between two audio signals. '''

        # Ensure the sample rates are the same
        if audio1.rate != audio2.rate:
            logger.handle_error(
                msg=f"Sample rates must be the same, got {audio1.rate} and {audio2.rate}",
                exception=ValueError
            )

        # Compute the offset between the two audio signals
        correlation = correlate(audio1.data, audio2.data, mode='full')
        lag = np.argmax(correlation) - (len(audio2) - 1)

        # Compute offset in seconds
        offset_time = lag / audio1.rate

        return abs(float(offset_time))

    @staticmethod
    def plot_aligned_audio(
        audio1    : AudioFile,
        audio2    : AudioFile,
        offset    : float = 0,
        title     : str   = '',
        save_path : str   = '',
        logger    : BaseLogger = SilentLogger()
    ):
        ''' Plots the aligned audio signals by applying an offset. '''
        
        # Ensure the sample rates are the same
        if audio1.rate != audio2.rate:
            logger.handle_error(
                msg=f"Sample rates must be the same, got {audio1.rate} and {audio2.rate}",
                exception=ValueError
            )
        
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
            plt.tight_layout()
            plt.savefig(save_path)

    @staticmethod
    def trim_video(
        video1    : VideoFile, 
        video2    : VideoFile,
        sync_time : float,
        out_path  : str, 
        logger    : BaseLogger = SilentLogger()
    ) -> Tuple[VideoFile, VideoFile]:
        ''' Trims the video signals to the same length by applying an offset derived from the sync time. '''

        # Use longest video as `video1`
        if video1.metadata.duration < video2.metadata.duration:
            video1, video2 = video2, video1

        # Get frame rates
        fps = min(video1.metadata.frame_rate, video2.metadata.frame_rate)

        # Calculate trimmed duration
        aligned_duration = min(
            video1.metadata.duration - sync_time, 
            video2.metadata.duration
        )

        out_video_paths = []

        for video, sync in zip([video1, video2], [sync_time, 0]):

            logger.info(f'Converting video {video.name}')

            camera_name = PathUtils.get_folder_name(video.path)
            out_video_path = os.path.join(out_path, f'{camera_name}.{VideoSync.VIDEO_EXT}')
            out_video_paths.append(out_video_path)

            cmd = (
                ffmpeg
                .input(video.path, ss=sync) 
                .filter('fps', fps=fps)   
                .output(out_video_path, t=aligned_duration, vcodec='libx264', acodec='aac', format='mp4')
                .overwrite_output()
            )

            logger.info(f'Converting video at {video.path} to {out_video_path} for synchronization ...')

            try: 
                timer = Timer()
                cmd.run(overwrite_output=True, quiet=True)
                logger.info(f'Done in {timer}. ')
            
            except Exception as e:
                logger.error(f'Error while converting video: {e}')
                raise e
            
            logger.info(f'')

        logger.info(f'Reading trimmed videos ...')
        out_video_path_1, out_video_path_2 = out_video_paths
        video1_trimmed = VideoFile(path=out_video_path_1, logger=logger)
        video2_trimmed = VideoFile(path=out_video_path_2, logger=logger)

        return video1_trimmed, video2_trimmed
