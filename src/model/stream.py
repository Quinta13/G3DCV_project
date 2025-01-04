from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple, Iterable
from collections import Counter

import numpy as np
import cv2 as cv

from src.model.typing import Frame, Size2D
from src.utils.io_ import (
    PathUtils, InputSanitizationUtils as ISUtils,
    VideoFile
)
from src.utils.misc import default
from src.utils.io_ import SilentLogger
from src.utils.io_ import BaseLogger
from src.model.typing import Views

class VideoStream:

    def __init__(
        self, 
        path    : str,
        name    : str | None = None,
        logger  : BaseLogger = SilentLogger(),
        verbose : bool       = False
    ):
        ''' Initialize a video stream object from a video file. '''

        # Logging
        self._logger         : BaseLogger = logger
        self._logger_verbose : BaseLogger = logger if verbose else SilentLogger()
        self._is_verbose     : bool       = verbose
        
        ISUtils.check_input    (path=path, logger=self._logger_verbose)
        ISUtils.check_extension(path=path, logger=self._logger_verbose, ext=VideoFile.VIDEO_EXT)

        # Video
        self._path: str = path
        self._metadata: VideoFile.VideoMetadata = VideoFile.VideoMetadata.from_video_path(path=self.path, logger=self._logger, verbose=verbose)
        
        # Open the video stream
        self._video_capture : cv.VideoCapture = cv.VideoCapture(self.path)
        if not self._video_capture.isOpened():
            self._logger.handle_error(msg=f"Failed to open video stream at {self.path}", exception=FileNotFoundError)
        
        # File name, defaults to folder and file name
        self._name = default(name, PathUtils.get_folder_and_file(path=self.path))
    
    # --- PROPERTIES ---

    @property    
    def path(self) -> str: return self._path

    @property
    def name(self) -> str: return self._name

    @property
    def metadata(self) -> VideoFile.VideoMetadata: return self._metadata

    @property
    def views(self) -> List[str]: 

        black_frame = np.zeros((*self.metadata.size, 3), dtype=np.uint8)
        return list(self._process_frame(frame=black_frame, frame_id=-1).keys())

    @property
    def _str_name(self) -> str: return 'VideoStream'

    
    @property
    def _str_params(self) -> Dict[str, Any]:

        w, h = self.metadata.size

        return {
            'name'   : self.name,
            'frames' : len(self),
            'size'   : f'{w}x{h} pixels',
            'views'  : len(self.views)
        }

    # --- MAGIC METHODS ---
    
    def __str__(self) -> str: return f"{self._str_name}[{'; '.join(f'{k}: {v}' for k, v in self._str_params.items())}]"
    
    def __repr__(self) -> str: return str(self)

    def __len__(self) -> int: return self.metadata.frames
    """ Return the number of frames in the video. """

    def __iter__(self) -> Iterator[Views]: return self.iter_range(start=0, end=len(self), step=1)
    """ Iterate over all frames in the video. """
        
    def __getitem__(self, idx: int) -> Views:
        """ Get a specific frame from the video. """

        # Read the frame
        self._video_capture.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._video_capture.read()

        if ret: return self._process_frame(frame=frame, frame_id=idx)
        else:   self._logger.handle_error(msg=f"Error: Unable to read frame at index {idx}.", exception=IndexError)
    
    # --- STREAMING ---

    def iter_range(self, start: int, end: int, step: int = 1) -> Iterator[Views]:
        """ Iterate over a specific range of frames in the video. """
        
        # Set the video position to the starting frame
        self._video_capture.set(cv.CAP_PROP_POS_FRAMES, start)

        frame_id = start

        while frame_id < end and self._video_capture.isOpened():
        
            # Skip frames
            for _ in range(step-1): 
                self._video_capture.grab()
                frame_id += 1

            # Read the frame
            ret, frame = self._video_capture.read()

            # Check if the frame was read successfully
            if not ret: break

            # Process the frame if required
            frame_ = self._process_frame(frame=frame, frame_id=frame_id)

            frame_id += 1

            yield frame_
    
    def play(
        self, 
        start        : int         = 0,
        end          : int  | None = None, 
        skip_frames  : int         = 1,
        window_size  : Size2D | None = None,
        exclude_views: List[str]   = []
    ):
        ''' 
        Stream the video from start to end frame. 
        It is possible to resize the window by specifying the `window_size` parameter.
        '''
        
        single_sync_stream = SynchronizedVideoStream(streams=[self], logger=self._logger, verbose=self._is_verbose)
        single_sync_stream.play(start=start, end=end, skip_frames=skip_frames, window_size=window_size, exclude_views={self.name: exclude_views})
    
    def _process_frame(self, frame: Frame, frame_id: int) -> Dict[str]:
        """
        Process a frame before displaying it.
        NOTE: This method can be overridden by subclasses to apply custom processing.
        """

        return {'raw': frame}


class SynchronizedVideoStream:
    
    EXIT_KEY = ord('q')

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
        start          : int                         = 0,
        end            : int | None                  = None, 
        skip_frames    : int                         = 1,
        window_size    : Size2D | List[Size2D] | None    = None,
        delay          : float | None                = None,
        exclude_views : Dict[str, List[str]] | None = None
    ):
        ''' 
        Stream the videos from start to end frame in a synchronized manner.
        Each video will be displayed in its own window.
        '''

        delay_        : float                = default(delay,          self.delay)
        end_          : int                  = default(end,            self._num_frames)
        exclude_views_: Dict[str, List[str]] = default(exclude_views, {})

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
                for view in stream.views:
                    if view in exclude_views_.get(stream.name, []): continue
                    name = f'{stream.name} - {view}'
                    cv.namedWindow (name, cv.WINDOW_NORMAL)
                    cv.resizeWindow(name, w, h)

        # Create iterators for each video stream
        iterators = [
            stream.iter_range(start=start, end=end_, step=skip_frames) 
            for stream in self._streams
        ]


        try:

            # Playback loop
            while True:

                # Read frames from all iterators
                for stream, iterator in zip(self._streams, iterators):
                    try:
                        frame_views = next(iterator) # type: ignore
                        for view, frame in frame_views.items():
                            if view in exclude_views_.get(stream.name, []): continue
                            cv.imshow(f'{stream.name} - {view}', frame)
                    except StopIteration:
                        # If any stream is finished, exit playback
                        cv.destroyAllWindows()
                        return

                # Wait for delay or exit on key press
                if cv.waitKey(int(delay_)) & 0xFF == SynchronizedVideoStream.EXIT_KEY:
                    break

            # Close all windows
            cv.destroyAllWindows()
        
        except Exception as e:
            self._logger.error(msg=f"Error occurred during video playback: {e}")
            cv.destroyAllWindows()
            raise e
