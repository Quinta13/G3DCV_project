from __future__ import annotations

import os
from typing import Dict, Tuple
from dataclasses import dataclass

import numpy as np
import cv2
import ffmpeg
from numpy.typing import NDArray
from scipy.io.wavfile import read as wav_read

from src.utils.io_ import SilentLogger, FormatLogger
from src.utils.io_ import InputSanitization as IS, PathOperations as PO


class AudioFile:

    def __init__(self, path: str, logger: FormatLogger = SilentLogger()):

        self._logger = logger

        IS.check_input(path=path, logger=self._logger)
        IS.check_extension(path=path, ext='.wav')

        self._path = path
        
        self._rate: int
        self._data: NDArray
        self._rate, self._data = self._read_audio()

    def _read_audio(self) -> Tuple[int, NDArray]:

        rate, data = wav_read(filename=self.path)

        # Convert to mono
        if data.ndim > 1: data = data.mean(axis=1)

        return rate, data
    
    def __str__(self) -> str:
        return f"Audio[{self.name}, "\
            f"{int  (self.duration)} sec,"\
            f"{round(self.rate, 2 )} Hz]"
    
    def __repr__(self) -> str: return str(self)

    def __len__(self) -> int: return len(self.data)

    @property
    def path(self) -> str: return self._path

    @property
    def name(self) -> str: return PO.get_file(self.path)

    @property
    def rate(self) -> int : return self._rate

    @property
    def data(self) -> NDArray: return self._data

    @property
    def duration(self) -> float: return len(self.data) / self.rate


class VideoFile:

    @dataclass
    class Metadata:
        
        ext          : str
        frames       : int
        duration     : float
        frame_rate   : float
        _sample_rate : int | None

        def __str__(self) -> str:
            return f"Metadata[ext: {self.ext}, "\
                f"{self.frames     } frames, "\
                f"{self.duration   } seconds, "\
                f"{self.frame_rate } fps, " +\
                (f"{self.sample_rate} Hz" if self.has_audio else "No audio") + "]"
        
        def __repr__(self) -> str: return str(self)
        
        @property
        def has_audio(self) -> bool: return self._sample_rate is not None
        
        @property
        def sample_rate(self) -> int:
            if self._sample_rate is None: raise ValueError(f"No audio stream found in video")
            return self._sample_rate

        def to_dict(self) -> Dict:

            return {
                'ext'         : self.ext,
                'num_frames'  : self.frames,
                'duration'    : self.duration,
                'frame_rate'  : self.frame_rate,
                'sample_rate' : self.sample_rate
            }

    VIDEO_EXT = {'.mov', '.mp4', '.avi'}

    def __init__(self, path, logger: FormatLogger = SilentLogger()):

        self._logger = logger

        IS.check_input(path=path, logger=self._logger)
        IS.check_extension(path=path, ext=VideoFile.VIDEO_EXT)

        self._path     = path
        self._metadata = self._load_metadata()
    
    def __str__(self) -> str:
        return f"Video[{self.name}, "\
            f"{int  (self.metadata.duration     )} sec, "\
            f"{round(self.metadata.frame_rate, 2)} fps]"
    
    def __repr__(self) -> str:
        return str(self)
    
    def __len__(self) -> int:
        return self.metadata.frames
    
    @property
    def name(self) -> str: return PO.get_file(self.path)

    @property
    def path(self) -> str: return self._path

    @property
    def audio_path(self) -> str: return f"{os.path.splitext(self.path)[0]}.wav"

    @property
    def has_audio(self) -> bool: return self.metadata.has_audio

    @property
    def metadata(self) -> Metadata: return self._metadata

    def _load_metadata(self) -> Metadata:

        self._logger.info(f"Loading metadata...")

        probe = ffmpeg.probe(self.path)
    
        # Video stream
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if not video_stream: raise ValueError(f"No video stream found in video {self.path}")
        
        # Audio stream
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        if not audio_stream: self._logger.warning(f"No audio stream found in video {self.path}")
        
        # Metadata
        ext         = os.path.splitext(self.path)[1].lower()
        num_frames  = int  (video_stream.get('nb_frames',      0))
        duration    = float(video_stream.get('duration',       0))
        frame_rate  = eval (video_stream.get('avg_frame_rate', 0))
        sample_rate = int  (audio_stream.get('sample_rate',    0)) if audio_stream else None
        
        return VideoFile.Metadata(
            ext          = ext,
            frames       = num_frames, 
            duration     = duration, 
            frame_rate   = frame_rate, 
            _sample_rate = sample_rate
        )

    def extract_audio_track(self) -> AudioFile:

        if not self.has_audio: 
            self._logger.error(f"No audio stream found in video {self.path}")
            raise ValueError(f"No audio stream found in video {self.path}")

        self.extract_audio(video=self, out_path=self.audio_path, logger=self._logger)
        return AudioFile(path=self.audio_path, logger=self._logger)
    
    @staticmethod
    def extract_audio(
        video: 'VideoFile',
        out_path: str,
        logger: FormatLogger = SilentLogger(),
        sample_rate: int | None = None
    ) -> AudioFile:
        
        IS.check_output(path=out_path, logger=logger)

        logger.info(f"Extracting audio from {video.path} to {out_path}...")

        try:

            (
                ffmpeg
                .input(video.path)
                .output(out_path, **({'ar': sample_rate} if sample_rate else {}))
                .run(overwrite_output=True)
            )

            logger.info(f"Audio extracted successfully")
            return AudioFile(path=out_path, logger=logger)

        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            raise e

    def close_audio(self):

        try:
            if os.path.exists(self.audio_path): 
                self._logger.info(f"Removing audio file {self.audio_path}")
                os.remove(self.audio_path)
        except Exception as e:
            self._logger.error(f"Failed to remove audio file: {e}")
            raise e 
    
    def __del__(self): self.close_audio()
