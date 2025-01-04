from __future__ import annotations

from abc import abstractmethod
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, Set, Tuple, Type

import ffmpeg
from loguru import logger as loguru_logger
from numpy.typing import NDArray
from scipy.io.wavfile import read as wav_read

from src.utils.misc import Timer


# _______________________________ LOGGER _______________________________

class BaseLogger(logging.Logger):
    ''' 
    Abstract class for loggers with a formatting function for info, warning and error channels.
    The formatting is implemented as a function that transform an input string into an output string.
    '''

    def __init__(self, name: str):
        ''' Initialize the logger with a name. '''

        super().__init__(name=name)

        # The default formatter is the identity function
        self._formatter: Callable[[str], str] = lambda x: x

    # --- FORMATTER ---

    @property
    def formatter(self) -> Callable[[str], str]: return self._formatter

    @formatter.setter
    def formatter(self, prefix: Callable[[str], str]): self._formatter = prefix

    def reset_formatter(self): self._formatter = lambda x: x

    # --- LOGGING CHANNELS ---

    ''' The pubic methods are predefined with the formatter function. '''
    def info   (self, msg): [self._info   (self.formatter(msg_)) for msg_ in msg.split("\n")]
    def warning(self, msg): [self._warning(self.formatter(msg_)) for msg_ in msg.split("\n")]
    def error  (self, msg): [self._error  (self.formatter(msg_)) for msg_ in msg.split("\n")]

    ''' The private methods are abstract and must be implemented by the subclasses. '''
    @abstractmethod
    def _info   (self, msg): raise NotImplementedError

    @abstractmethod
    def _warning(self, msg): raise NotImplementedError

    @abstractmethod
    def _error  (self, msg): raise NotImplementedError

    def handle_error(self, msg: str, exception: Type[Exception] = ValueError):
        ''' Log the error message and raise an exception. '''

        self.error(msg)
        raise exception(msg)

class SilentLogger(BaseLogger):
    ''' Logger implemented as a no-op. '''

    def __init__(self): super().__init__(name='SilentLogger')

    def _info   (self, msg): pass
    def _warning(self, msg): pass
    def _error  (self, msg): pass


class PrintLogger(BaseLogger):
    ''' Logger that prints messages to the console. '''

    def __init__(self): super().__init__(name='PrintLogger')

    def _info   (self, msg): print(f"INFO:  {msg}")
    def _warning(self, msg): print(f"WARN:  {msg}")
    def _error  (self, msg): print(f"ERROR: {msg}")


class FileLogger(BaseLogger):
    ''' Logger that writes messages to a .log file. '''

    def __init__(self, file, level=logging.INFO):

        super().__init__(name='FileLogger')

        InputSanitizationUtils.check_extension(path=file, ext='.log')

        loguru_logger.add(file, level=level)
        self._logger = loguru_logger

    def _info   (self, msg): self._logger.info   (msg)
    def _warning(self, msg): self._logger.warning(msg)
    def _error  (self, msg): self._logger.error  (msg)


# _______________________________ CONTAINER UTILS CLASSES _______________________________

class PathUtils:

    @staticmethod
    def get_folder_path(path: str) -> str: return os.path.dirname(path)
    ''' Get the folder path of the file (e.g. /path/to/file.txt -> /path/to). '''

    @staticmethod
    def get_folder_name(path: str) -> str: return os.path.basename(PathUtils.get_folder_path(path))
    ''' Get the name of file containing folder (e.g. /path/to/file.txt -> to). '''

    @staticmethod
    def get_file(path: str) -> str: return os.path.basename(path)
    ''' Get the file from the path (e.g. /path/to/file.txt -> file.txt). '''

    @staticmethod
    def get_folder_and_file(path: str) -> str: return os.path.join(PathUtils.get_folder_name(path), PathUtils.get_file(path))
    ''' Get the folder and file from the path (e.g. /path/to/file.txt -> /to/file.txt). '''

    @staticmethod
    def get_file_name(path: str) -> str:
        ''' Get the file name from the path (e.g. /path/to/file.txt -> file). '''

        file, ext = os.path.splitext(PathUtils.get_file(path))
        return file
    
    @staticmethod
    def get_file_ext(path: str) -> str:
        ''' Get the file extension from the path (e.g. /path/to/file.txt -> txt). '''

        file, ext = os.path.splitext(PathUtils.get_file(path))
        return ext[1:].lower() # Remove the dot
    
class IOUtils:

    @staticmethod
    def make_dir(path: str, logger: BaseLogger = SilentLogger()):
        ''' Create a directory if it does not exist. '''

        if not os.path.exists(path): 
            os.makedirs(path)
            logger.info(msg=f"Directory created at: {path}")
        else:
            logger.info(msg=f"Directory already found at: {path}")
    
    @staticmethod
    def error_handler(msg: str, logger: BaseLogger, exception: Type[Exception] = ValueError):
        ''' Log the error message and raise an exception. '''

        logger.error(msg)
        raise exception(msg)


class InputSanitizationUtils:

    @staticmethod
    def check_input(path: str, logger: BaseLogger = SilentLogger()):
        ''' Check if input file exists. '''

        if not os.path.exists(path):  
            logger.handle_error(
                msg=f"Input file not found: {path}", 
                exception=FileNotFoundError
            )
        else:
            logger.info(msg=f"Input file found at: {path} ")

    @staticmethod
    def check_output(path: str, logger: BaseLogger = SilentLogger()):
        ''' Check if the directory of the output file exists. '''

        out_dir = PathUtils.get_folder_path(path)

        if not os.path.exists(out_dir): 
            logger.handle_error(
                msg=f"Output directory not found: {out_dir}", 
                exception=FileNotFoundError
            )
        else:                           
            logger.info(msg=f"Output directory found at: {out_dir} ")

    @staticmethod
    def check_extension(path: str, ext: str | Set[str], logger: BaseLogger = SilentLogger()):
        ''' Check if any of the extensions in the list match the file extension. '''

        if type(ext) == str: ext = {ext}

        if not any([path.endswith(e) for e in f'.{ext}']):

            logger.handle_error(
                msg=f"Invalid file extension: {path}. Expected one of {ext} extensions.", 
                exception=ValueError
            )


# _______________________________ AUDIO & VIDEO FILES _______________________________


class AudioFile:
    ''' Class to handle audio files. '''

    EXT_AUDIO = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'wma', 'aac', 'aiff', 'au', 'raw'}

    # --- INITIALIZATION ---

    def __init__(self, path: str, logger: BaseLogger = SilentLogger(), verbose: bool = False):
        ''' Initialize the audio file with its path '''

        self._logger        : BaseLogger = logger
        self._logger_verbose: BaseLogger = logger if verbose else SilentLogger()
        self._is_verbose    : bool = verbose

        self._path: str = path

        self._rate: int
        self._data: NDArray
        self._rate, self._data = self._read_audio()

    def _read_audio(self) -> Tuple[int, NDArray]:
        ''' Read the audio file and return the sample rate and data. '''

        InputSanitizationUtils.check_input    (path=self.path, logger=self._logger_verbose)
        InputSanitizationUtils.check_extension(path=self.path, logger=self._logger_verbose, ext=self.EXT_AUDIO)

        self._logger.info(msg=f"Reading audio file from {self.path} ...")

        timer = Timer()

        try: 
            rate, data = wav_read(filename=self.path)
            self._logger.info(msg=f"Audio read successfully in {timer}.")
        except Exception as e:
            self._logger.error(f"Failed to extract audio: {e}")
            raise e

        # Convert to mono
        if data.ndim > 1: data = data.mean(axis=1)

        return rate, data
    
    # --- MAGIC METHODS ---

    def __str__(self) -> str:

        return f"Audio[{self.name}; "\
            f"duration: {int  (self.duration)} sec; "\
            f"sampling rate: {round(self.rate, 2 )} Hz]"

    def __repr__(self) -> str: return str(self)

    def __len__(self) -> int: return len(self.data)
    ''' Number of samples in the audio file. '''

    # --- PROPERTIES ---

    @property
    def path(self) -> str: return self._path

    @property
    def name(self) -> str: return PathUtils.get_folder_and_file(path=self.path)

    @property
    def rate(self) -> int: return self._rate
    ''' Sampling rate of the audio file in Hz. '''

    @property
    def data(self) -> NDArray: return self._data
    ''' Data of the audio file as a one dimensional numpy array. '''

    @property
    def duration(self) -> float: return len(self.data) / self.rate
    ''' Duration of the audio file in seconds. '''


class VideoFile:
    ''' Class to handle video files. '''

    @dataclass
    class VideoMetadata:
        ''' Dataclass to store video metadata. '''

        ext          : str             # File extension  
        frames       : int             # Number of video frames
        duration     : float           # Video duration in seconds
        fps          : float           # Video frame rate in frames per second
        size         : Tuple[int, int] # Video frame size
        _sample_rate : int | None      # Audio sample rate, if audio stream is present

        @classmethod
        def from_dict(cls, data: Dict) -> 'VideoFile.VideoMetadata':
            ''' Create video metadata from a dictionary. '''

            try:
                return cls(
                    ext          = data['ext'],
                    frames       = data['num_frames'],
                    duration     = data['duration'],
                    fps          = data['frame_rate'],
                    size         = data['size'],
                    _sample_rate = data['sample_rate']
                )

            except KeyError as e:
                raise ValueError(f"Invalid dictionary for video metadata: {e}")

        @classmethod
        def from_video_path(cls, path: str, logger: BaseLogger, verbose: bool = False) -> 'VideoFile.VideoMetadata':
            ''' Load video metadata from a video file path. '''
            
            logger_verbose = logger if verbose else SilentLogger()
            
            InputSanitizationUtils.check_input    (path=path, logger=logger_verbose)
            InputSanitizationUtils.check_extension(path=path, logger=logger_verbose, ext=VideoFile.VIDEO_EXT)

            logger.info(msg=f"Loading metadata from video {path}")

            probe = ffmpeg.probe(path)

            # Video stream
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if not video_stream: logger.handle_error(msg=f"No video stream found in video {path}", exception=ValueError)

            # Audio stream
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            if not audio_stream: logger.warning(msg=f"No audio stream found in video {path}")

            # Metadata
            return cls(
                ext          = PathUtils.get_file_ext(path),
                frames       = int  (video_stream.get('nb_frames',      0)),
                duration     = float(video_stream.get('duration',       0)),
                fps          = eval (video_stream.get('avg_frame_rate', 0)),
                _sample_rate = int  (audio_stream.get('sample_rate',    0)) if audio_stream else None,
                size         = (
                    int(video_stream.get('width',  0)), 
                    int(video_stream.get('height', 0))
                ),
            )

        def __str__(self) -> str:

            return f"Metadata[ext: {self.ext}; "\
                f"{self.frames     } frames; "\
                f"{self.duration   } seconds; "\
                f"{self.fps } fps; "\
                f"{self.size       } size; " +\
                (f"{self.sample_rate} Hz" if self.has_audio else "No audio") + "]"

        def __repr__(self) -> str: return str(self)

        @property
        def has_audio(self) -> bool: return self._sample_rate is not None

        @property
        def sample_rate(self) -> int:
            if not self.has_audio: raise ValueError(f"No audio stream found in video.")
            return self._sample_rate  # type: ignore - Not none check is done in the property

        def to_dict(self) -> Dict:

            return {
                'ext'         : self.ext,
                'num_frames'  : self.frames,
                'duration'    : self.duration,
                'frame_rate'  : self.fps,
                'size'        : self.size,
                'sample_rate' : self.sample_rate,
            }

    VIDEO_EXT = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'webm', 'wmv', 'mpg', 'ogv'}
    AUDIO_EXTRACTION_EXT = 'wav'

    # --- INITIALIZATION ---

    def __init__(self, path, logger: BaseLogger = SilentLogger(), verbose: bool = False):
        ''' Initialize the video file with its path '''

        self._logger        : BaseLogger = logger
        self._logger_verbose: BaseLogger = logger if verbose else SilentLogger()
        self._is_verbose    : bool = verbose

        self._path: str = path

        self._metadata: VideoFile.VideoMetadata = VideoFile.VideoMetadata.from_video_path(
            path=self.path,
            logger=self._logger, 
            verbose=self._is_verbose
        )
    
    # --- MAGIC METHODS ---

    def __str__(self) -> str:

        w, h = self.metadata.size
        
        return f"Video[{self.name}; "\
            f"duration: {int  (self.metadata.duration     )} sec; "\
            f"size: {w}x{h} pixels; "\
            f"frame rate: {round(self.metadata.fps, 2)} fps]"

    def __repr__(self) -> str: return str(self)

    def __len__(self) -> int: return self.metadata.frames
    ''' Number of frames in the video file. '''

    # --- PROPERTIES ---

    @property
    def name(self) -> str: return PathUtils.get_folder_and_file(path=self.path)

    @property
    def path(self) -> str: return self._path

    @property
    def metadata(self) -> VideoMetadata: return self._metadata

    @property
    def has_audio(self) -> bool: return self.metadata.has_audio
    ''' Check if the video file has an audio stream. '''

    @property
    def audio_path(self) -> str: 
        ''' 
        Path to the extracted audio file.
        That is the same as the video file with audio extension.
        '''
        return os.path.join(PathUtils.get_folder_path(self.path), f'{PathUtils.get_file_name(self.path)}.{self.AUDIO_EXTRACTION_EXT}')

    # --- AUDIO EXTRACTION ---

    '''
    NOTE:   When working on audio file, we extract the audio track from the video file in the same directory.
            The audio file is deleted when the video file object is deleted.
    '''

    def extract_audio_track(self) -> AudioFile:
        ''' Extract the audio track from the video file and save it to the same folder with the video file. '''

        if not self.has_audio:
            self._logger.handle_error(f"Cannot extract audio stream in video {self.path}", exception=ValueError)

        audio = self.extract_audio(video=self, out_path=self.audio_path, logger=self._logger, verbose=self._is_verbose)
        self._logger.info(msg='Use method `close_audio` to remove the audio file from the file system.')
        return audio

    @staticmethod
    def extract_audio(
        video: 'VideoFile',
        out_path: str,
        sample_rate: int | None = None,
        logger: BaseLogger = SilentLogger(),
        verbose: bool = False
    ) -> AudioFile:
        '''
        Extract the audio track from the video file and save it to the output path.
        It optionally resamples the audio to the given sample rate.
        '''

        verbose_logger = logger if verbose else SilentLogger()

        InputSanitizationUtils.check_output(path=out_path, logger=verbose_logger)

        logger.info(msg=f"Extracting audio from {video.path} to {out_path} ...")

        try:

            timer = Timer()

            # Add sample rate argument if provided
            sample_rate_args = {'ar': sample_rate} if sample_rate else {}

            cmd = (
                ffmpeg
                .input(video.path)
                .output(out_path, **sample_rate_args)
                .overwrite_output()
            )
            
            cmd.run(overwrite_output=True, quiet=True)

            logger.info(msg=f"Audio extracted successfully in {timer}.")
            return AudioFile(path=out_path, logger=logger)

        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            raise e

    def close_audio(self):
        '''
        Close the audio file by removing it from the file system.
        '''

        try:

            if os.path.exists(self.audio_path):
                self._logger.info(msg=f"Removing audio file {self.audio_path}")
                os.remove(self.audio_path)
            else:
                self._logger.warning(msg=f"Cannot remove audio file at: {self.audio_path}, not found")

        except Exception as e:
            self._logger.error(f"Failed to remove audio file: {e}")
            raise e

    def __del__(self): 
        ''' Remove the audio if it was extracted when the video file object is deleted '''
        self._logger = SilentLogger() # Prevent logging
        self.close_audio()
