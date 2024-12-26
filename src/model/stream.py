from __future__ import annotations

from typing import Any, Dict, List, Tuple, Iterable
from collections import Counter

from numpy.typing import NDArray
import cv2 as cv

from src.utils.io_ import (
    BaseLogger, SilentLogger,
    PathUtils, InputSanitizationUtils as ISUtils,
    VideoFile
)
from src.utils.misc import default, Size

class SynchronizedVideoStream:

    def __init__(self, streams: List[VideoStream], logger: BaseLogger = SilentLogger(), verbose: bool = False):
        ''' Initialize a synchronized video stream object from multiple video streams. '''

        self._logger        : BaseLogger = logger
        self._logger_verbose: BaseLogger = logger if verbose else SilentLogger()
        self._is_verbose    : bool = verbose

        # Check at least one video stream
        if not streams:
            self._logger.handle_error(msg="At least one video stream must be provided.", exception=ValueError)

        # Check if all streams have the same number of frames
        self._streams: List[VideoStream] = streams
        frames = {len(stream) for stream in self._streams}
        if len(frames) > 1:
            self._logger.handle_error(
                msg=f"All video streams must have the same number of frames. Got multiple: {frames}",
                exception=ValueError
            )
        
        # Check all different names
        repeated_names = {k for k, c in Counter([s.name for s in self._streams]).items() if c > 1}
        repeated_names = {}
        if repeated_names:
            self._logger.handle_error(
                msg=f"Video stream names must be unique. Got repeated names: {repeated_names}",
                exception=ValueError
            )
        
        self._num_frames: int = frames.pop()
    
    def __len__(self) -> int: return len(self._streams)
    
    def __str__(self) -> str: 
        
        stream_names = ', '.join(stream.name for stream in self._streams)
        return f"SynchronizedVideoStream[{len(self)} streams ({stream_names}); frames: {self._num_frames}]"
    
    def __repr__(self) -> str: return str(self)

    @property
    def delay(self) -> float: return 1000 / self._streams[0].metadata.fps

    @property
    def num_frames(self) -> int: return self._num_frames

    def play(
        self,
        start      : int                                            = 0,
        end        : int | None                                     = None, 
        skip_frames: int                                            = 1,
        window_size: Size | List[Size] | None = None,
        delay      : float | None                                   = None
    ):
        ''' 
        Stream the videos from start to end frame in a synchronized manner.
        Each video will be displayed in its own window.
        '''

        delay_: float = default(delay, self.delay)
        end_  : int   = default(end, self._num_frames)

        # Window size
        if window_size is not None:

            if isinstance(window_size, tuple): window_size = [window_size] * len(self._streams)

            else:
                if len(window_size) != len(self._streams):
                    self._logger.handle_error(
                        msg=f"Number of window sizes must be equal to number of video streams. "
                            f"Got {len(window_size)} and {len(self._streams)}",
                        exception=ValueError
                    )

            for (w, h), stream in zip(window_size, self._streams):
                cv.namedWindow (stream.name, cv.WINDOW_NORMAL)
                cv.resizeWindow(stream.name, w, h)

        # Create iterators for each video stream
        iterators = [
            stream.iter_range(start=start, end=end_, step=skip_frames, preprocess=True) 
            for stream in self._streams
        ]

        try:

            # Playback loop
            while True:

                # Read frames from all iterators
                for stream, iterator in zip(self._streams, iterators):
                    try:
                        frame = next(iterator)               # type: ignore
                        cv.imshow(stream.name, frame)
                    except StopIteration:
                        # If any stream is finished, exit playback
                        cv.destroyAllWindows()
                        return

                # Wait for delay or exit on key press
                if cv.waitKey(int(delay_)) & 0xFF == VideoStream.EXIT_KEY:
                    break

            # Close all windows
            cv.destroyAllWindows()
        
        except Exception as e:
            self._logger.error(msg=f"Error occurred during video playback: {e}")
            cv.destroyAllWindows()
            raise e



class VideoStream:

    EXIT_KEY = ord('q')

    def __init__(self, path: str, name: str = '', logger: BaseLogger = SilentLogger(), verbose: bool = False):
        ''' Initialize a video stream object from a video file. '''

        self._logger        : BaseLogger = logger
        self._logger_verbose: BaseLogger = logger if verbose else SilentLogger()
        self._is_verbose    : bool = verbose
        
        ISUtils.check_input    (path=path, logger=self._logger_verbose)
        ISUtils.check_extension(path=path, logger=self._logger_verbose, ext=VideoFile.VIDEO_EXT)

        self._path: str = path
        self._metadata: VideoFile.VideoMetadata = VideoFile.VideoMetadata.from_video_path(path=self.path, logger=self._logger, verbose=verbose)

        self._video_capture = cv.VideoCapture(self.path)

        if not self._video_capture.isOpened():
            self._logger.handle_error(msg=f"Failed to open video stream at {self.path}", exception=FileNotFoundError)
        
        self._name = name if name else PathUtils.get_folder_and_file(path=self.path)
    
    # --- PROPERTIES ---

    @property    
    def path(self) -> str: return self._path

    @property
    def metadata(self) -> VideoFile.VideoMetadata: return self._metadata

    @property
    def name(self) -> str: return self._name

    # --- MAGIC METHODS ---
    
    @property
    def _str_name(self) -> str: return 'VideoStream'
    
    @property
    def _str_params(self) -> Dict[str, Any]:

        w, h = self.metadata.size

        return {
            'name'    : self.name,
            'frames'  : len(self),
            'size'    : f'{w}x{h} pixels',
        }

    def __str__(self) -> str: return f"{self._str_name}[{'; '.join(f'{k}: {v}' for k, v in self._str_params.items())}]"
    
    def __repr__(self) -> str: return str(self)

    def __len__(self) -> int: return self.metadata.frames
    """ Return the number of frames in the video. """

    def __iter__(self) -> Iterable[NDArray]: return self.iter_range(start=0, end=len(self), step=1, preprocess=True)
    """ Iterate over all frames in the video. """
        
    def __getitem__(self, idx: int) -> NDArray:
        """ Get a specific frame from the video. """

        # Read the frame
        self._video_capture.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._video_capture.read()

        if ret: return self._process_frame(frame=frame, frame_id=idx)
        else:   self._logger.handle_error(msg=f"Error: Unable to read frame at index {idx}.", exception=IndexError)
    
    # --- STREAMING ---

    def iter_range(self, start: int, end: int, step: int = 1, preprocess: bool = True) -> Iterable[NDArray]:
        """ Iterate over a specific range of frames in the video. """
        
        # Set the video position to the starting frame
        self._video_capture.set(cv.CAP_PROP_POS_FRAMES, start)

        frame_id = start

        while frame_id < end and self._video_capture.isOpened():
        
            # Skip frames
            for _ in range(step-1): self._video_capture.grab(); frame_id += 1

            # Read the frame
            ret, frame = self._video_capture.read()

            # Check if the frame was read successfully
            if not ret: break

            # Process the frame if required
            frame_ = frame if not preprocess else self._process_frame(frame=frame, frame_id=frame_id)

            frame_id += 1

            yield frame_
    
    def play(
        self, 
        start      : int         = 0,
        end        : int  | None = None, 
        skip_frames: int         = 1,
        window_size: Size | None = None
    ):
        ''' 
        Stream the video from start to end frame. 
        It is possible to resize the window by specifying the `window_size` parameter.
        '''
        
        single_sync_stream = SynchronizedVideoStream(streams=[self], logger=self._logger, verbose=self._is_verbose)
        single_sync_stream.play(start=start, end=end, skip_frames=skip_frames, window_size=window_size)
    
    def _process_frame(self, frame: NDArray, frame_id: int) -> NDArray:
        """
        Process a frame before displaying it.
        NOTE: This method can be overridden by subclasses to apply custom processing.
        """

        return frame


