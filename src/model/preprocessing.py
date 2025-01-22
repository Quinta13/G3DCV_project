import math
import os
import numpy as np
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.signal import correlate
import ffmpeg
import cv2 as cv


from src.utils.typing import Size2D
from src.utils.calibration import CalibratedCamera
from src.utils.stream import VideoStream
from src.utils.io_ import (
    PathUtils, AudioFile, VideoFile
)
from src.utils.misc import Timer
from src.utils.io_ import SilentLogger
from src.utils.io_ import BaseLogger


class VideoSync:

    AUDIO_EXT = 'wav'

    def __init__(
        self, 
        video1        : VideoFile, 
        video2        : VideoFile,
        out_dir       : str, 
        out_video_ext : str        = 'mp4',
        logger        : BaseLogger = SilentLogger(),
        verbose       : bool       = False
    ):
        '''
        Receive in input two video files to synchronize and an output directory to save the synchronization data.
        '''

        # Logger
        self._logger        : BaseLogger = logger
        self._logger_verbose: BaseLogger = logger if verbose else SilentLogger()
        self._is_verbose    : bool       = verbose

        # Check if the videos refer to the same experiment - i.e. have the same name
        exp1 = PathUtils.get_file_name(video1.path)
        exp2 = PathUtils.get_file_name(video2.path)

        if exp1 != exp2:
            logger.handle_error(
                msg=f"Experiments must have the same name as they refer to the same experiment, "\
                    f"got {exp1} and {exp2}. Please check input videos match the same experiment.",
                exception=ValueError
            )
        
        self._video1        : VideoFile = video1
        self._video2        : VideoFile = video2
        self._exp_name      : str       = exp1
        self._out_video_ext : str       = out_video_ext
        self._out_dir       : str       = out_dir

    # --- MAGIC METHODS ---

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{self.exp_name}]"

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

    def sync(self) -> Tuple[VideoStream, VideoStream]:

        timer = Timer()

        # Synchronization process
        self._logger.info(msg=f"SYNCING VIDEOS: ")
        self._logger.info(msg=f" > 1. {self._video1}")
        self._logger.info(msg=f" > 2. {self._video2}")
        self._logger.info(msg=f"")

        # Extract audio from the videos
        self._logger.info(msg="EXTRACTING AUDIO FROM VIDEOS")
        audio1, audio2 = self._extract_audio()
        self._logger.info(msg='')
    
        # Calculate the offset between the two audio signals
        self._logger.info(msg="CALCULATING AUDIO SIGNALS OFFSET")
        offset = self.calculate_sync_offset(audio1=audio1, audio2=audio2)
        self._logger.info(msg=f"Calculated offset: {offset} seconds.")
        self._logger.info(msg='')

        # Plot audio signals before and after computed alignment
        self._logger.info(msg="PLOTTING AUDIO SIGNALS")
        for off, title in zip([0., offset], ['Original', 'Aligned']):

            self.plot_aligned_audio(
                audio1=audio1,
                audio2=audio2,
                offset=off,
                title=f'{self._exp_name} {title} Audio Signals',
                save_path=os.path.join(self.out_dir, f'audio-sync_{title.lower()}.png'),
                logger=self._logger
            )
        self._logger.info(msg='')
        
        # Save trimmed audios
        self._logger.info(msg="TRIMMING AUDIO SIGNALS")
        video1_out, video2_out = self.trim_video(
            video1=self._video1,
            video2=self._video2,
            sync_time=offset,
            out_video_ext=self._out_video_ext,
            out_path=self.out_dir,
            logger=self._logger
        )
        self._logger.info(msg='')

        # Log synchronization completion
        self._logger.info(msg=f"SYNC COMPLETED in {timer}.")
        self._logger.info(msg=f"")

        # Log synced videos
        self._logger.info(msg=f"SYNCED VIDEOS: ")
        self._logger.info(msg=f" > 1. {video1_out}")
        self._logger.info(msg=f" > 2. {video2_out}")

        frame_difference = video1_out.metadata.duration - video2_out.metadata.duration

        if frame_difference != 0:

            self._logger.warning(
                msg=f"After trimming videos have different number of frames: "\
                    f"{video1_out.metadata.frames} and {video2_out.metadata.frames}. "\
            )

        else :
            self._logger.info(msg=f"Trimmed videos have the same number of frames: {video1_out.metadata.frames}.")

        self._logger.info(msg=f"")

        return (
            VideoStream(path=video1_out.path),
            VideoStream(path=video2_out.path),
        )

    def _extract_audio(self) -> Tuple[AudioFile, AudioFile]:
        ''' Extract audio from the two videos with the same sample rate. '''

        # Compute the minimum sample rate
        self._logger.info(msg="Videos sampling rates: ")
        self._logger.info(msg=f" > 1. {self._video1.metadata.sample_rate} Hz")
        self._logger.info(msg=f" > 2. {self._video2.metadata.sample_rate} Hz")
        min_sample_rate = min(self._video1.metadata.sample_rate, self._video2.metadata.sample_rate)
        self._logger.info(msg=f"Extracting audio files from videos with minimum sample rate: {min_sample_rate}")

        # Extract audio for both the videos
        audios: List[AudioFile] = []

        for video in [self._video1, self._video2]:

            self._logger.info(msg=f"")
            self._logger.info(msg=f"Extracting {video.name} ...")
            self._logger.formatter = lambda x: f' - {x}'

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

            self._logger.reset_formatter()

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
            logger.info(msg=f"Saving plot to {save_path}")
            plt.tight_layout()
            plt.savefig(save_path)

    @staticmethod
    def trim_video(
        video1        : VideoFile, 
        video2        : VideoFile,
        sync_time     : float,
        out_path      : str, 
        out_video_ext : str        = 'mp4',
        logger        : BaseLogger = SilentLogger()
    ) -> Tuple[VideoFile, VideoFile]:
        ''' Trims the video signals to the same length by applying an offset derived from the sync time. '''

        # Use longest video as `video1`
        if video1.metadata.duration < video2.metadata.duration:
            video1, video2 = video2, video1

        # Get frame rates
        fps = min(video1.metadata.fps, video2.metadata.fps)

        # Calculate trimmed duration
        aligned_duration = min(
            video1.metadata.duration - sync_time, 
            video2.metadata.duration
        )

        out_video_paths = []

        for video, sync in zip([video1, video2], [sync_time, 0]):

            logger.info(msg=f'Converting video {video.name}')

            camera_name = PathUtils.get_folder_name(video.path)
            out_video_path = os.path.join(out_path, f'{camera_name}.{out_video_ext}')
            out_video_paths.append(out_video_path)

            cmd = (
                ffmpeg
                .input(video.path, ss=sync) 
                .filter('fps', fps=fps)   
                .output(out_video_path, t=aligned_duration, vcodec='libx264', acodec='aac', format='mp4')
                .overwrite_output()
            )

            logger.info(msg=f'Converting video at {video.path} to {out_video_path} for synchronization ...')

            try: 
                timer = Timer()
                cmd.run(overwrite_output=True, quiet=True)
                logger.info(msg=f'Done in {timer}. ')
            
            except Exception as e:
                logger.error(f'Error while converting video: {e}')
                raise e
            
            logger.info(msg=f'')

        logger.info(msg=f'Reading trimmed videos ...')
        out_video_path_1, out_video_path_2 = out_video_paths
        video1_trimmed = VideoFile(path=out_video_path_1, logger=logger)
        video2_trimmed = VideoFile(path=out_video_path_2, logger=logger)

        return video1_trimmed, video2_trimmed

class ChessboardCameraCalibrator(VideoStream):
    ''' Class to calibrate a camera using from video stream by detecting chessboard corners. '''

    def __init__(
            self, 
            path            : str, 
            chessboard_size : Size2D,
            samples         : int,
            logger          : BaseLogger = SilentLogger(), 
            verbose         : bool       = False
        ):
        '''
        The initialization requires the chessboard size and the number of samples to use for calibration.
        '''

        super().__init__(path=path, logger=logger, verbose=verbose)

        self._chessboard_size: Size2D = chessboard_size
        self._samples:         int  = samples

        self._reset_img_points()

    def _reset_img_points(self): self._img_points: List[NDArray] = []
    ''' Reset the collection of image points. '''

    @property
    def obj_point(self) -> NDArray:
        ''' Return the real world coordinates of the chessboard corners. '''

        h, w = self._chessboard_size

        # h x w x 3 matrix
        obj_point = np.zeros((h * w, 3), np.float32)

        # Grid coordinates for the chessboard (the square has a size of 1)
        # the z axis is 0 since the chessboard is flat
        obj_point[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)

        return obj_point
    
    @property
    def chessboard_size(self) -> Size2D: return self._chessboard_size

    @property
    def samples(self) -> int: return self._samples

    @property
    def skip_frames(self) -> int: return len(self) // self.samples
    ''' The sample frames are equally spaced in the video stream. '''
    
    def __str__(self) -> str:

        h, w = self.chessboard_size

        return f"{self.__class__.__name__}[{self.name}; "\
            f"chessboard size={h}x{w}; "\
            f"samples={self.samples}; "\
            f"skip frames={self.skip_frames}) "\
    
    def calibrate(self, window_size: Dict[str, Size2D] | Size2D | None = None) -> CalibratedCamera:
        ''' Calibrate the camera using the collected image points. '''

        self._reset_img_points()

        # Collect image points
        self._logger.info(msg=f"Collecting image point for camera using {self.samples} samples ... ")
        timer = Timer()
        self.play(start=self.skip_frames, skip_frames=self.skip_frames, window_size=window_size)
        self._logger.info(msg=f'Completed in {timer}. Collected samples: {len(self._img_points)}.')

        # Compute object points
        object_point = self.obj_point
        object_points = [object_point] * len(self._img_points)

        # Calibrate the camera
        return CalibratedCamera.from_points(
            obj_points=object_points,
            img_points=self._img_points,
            size=self.metadata.size,
            logger=self._logger,
            params={
                "chessboard_size" : self.chessboard_size,
                "samples"         : self.samples,
                "video_file"      : self.path,
            }
        )
    

    def _process_frame(self, frame: NDArray, frame_id: int) -> Dict[str, NDArray]:

        corners = self._find_chessboard_corners(frame=frame)

        if corners is not None:

            self._img_points.append(corners)
            frame_ = cv.drawChessboardCorners(frame.copy(), self._chessboard_size, corners, True)
    
        else:
            
            frame_ = frame
            if not self._is_debug(frame_id=frame_id): self._logger.warning(msg=f"[{self.name}] Unable to find chessboard in frame {frame_id}")

        return {'calibration': frame_} | super()._process_frame(frame=frame, frame_id=frame_id)
    
    def _find_chessboard_corners(self, frame: NDArray) -> NDArray | None:

        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        ret, corners = cv.findChessboardCorners(gray, self._chessboard_size, None)

        if ret:

            # Enhance the precision of the found corners
            # REF: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
            corners_refined = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            return corners_refined

        return None