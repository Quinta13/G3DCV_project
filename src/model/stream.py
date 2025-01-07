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
        name    : str        = '',
        logger  : BaseLogger = SilentLogger(),
        verbose : bool       = False
    ):

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
        if name == '': name = PathUtils.get_folder_and_file(path=self.path)
        self._name = name
    
    # --- PROPERTIES ---

    @property    
    def path(self) -> str: return self._path

    @property
    def name(self) -> str: return self._name

    @name.setter
    def name(self, value: str) -> None: self._name = value

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
        start        : int                               = 0,
        end          : int                        | None = None, 
        skip_frames  : int                               = 1,
        window_size  : Dict[str, Size2D] | Size2D | None = None,
        exclude_views: List[str]                         = []
    ):
        ''' 
        Stream the video from start to end frame. 
        It is possible to resize the window by specifying the `window_size` parameter.
        '''

        window_size_ = {self.name: window_size} if window_size is not None else dict()
        
        single_sync_stream = SynchronizedVideoStream(streams=[self], logger=self._logger, verbose=self._is_verbose)
        single_sync_stream.play(start=start, end=end, skip_frames=skip_frames, window_size=window_size_, exclude_views={self.name: exclude_views})

    def _is_debug(self, frame_id: int) -> bool: return frame_id < 0
    
    def _process_frame(self, frame: Frame, frame_id: int) -> Views:
        """
        Process a frame before displaying it.
        NOTE: This method can be overridden by subclasses to apply custom processing.
        """

        return {'raw': frame}


class SynchronizedVideoStream:
    
    EXIT_KEY  = 'q'
    VIEW_KEYS = '1234567890abcdefghijklmnoprstuvwxyz'

    def __init__(self, streams: List[VideoStream], logger: BaseLogger = SilentLogger(), verbose: bool = False):
        ''' Initialize a synchronized video stream object from multiple video streams. '''

        self._logger        : BaseLogger = logger
        self._logger_verbose: BaseLogger = logger if verbose else SilentLogger()
        self._is_verbose    : bool = verbose

        # Check at least one video stream
        if not streams:
            self._logger.handle_error(msg="At least one video stream must be provided.", exception=ValueError)

        # Check if all streams have the same number of frames
        frames = {len(stream) for stream in streams}
        if len(frames) > 1:
            self._logger.handle_error(
                msg=f"All video streams must have the same number of frames. Got multiple ({len(frames)}): {frames}",
                exception=ValueError
            )
        
        # Check all different names
        repeated_names = {k for k, c in Counter([s.name for s in streams]).items() if c > 1}
        if repeated_names:
            self._logger.handle_error(
                msg=f"Video stream names must be unique. Got repeated names: {repeated_names}",
                exception=ValueError
            )
        
        self._num_frames: int = frames.pop()

        # Save streams indexed by name
        self._streams: Dict[str, VideoStream] = {stream.name: stream for stream in streams}
    
    def __len__(self) -> int: return len(self._streams)
    
    def __str__(self) -> str: 
        
        stream_names = ', '.join(self._streams.keys())
        return f"SynchronizedVideoStream[{len(self)} streams ({stream_names}); frames: {self._num_frames}]"
    
    def __repr__(self) -> str: return str(self)

    @property
    def streams(self) -> List[VideoStream]: return list(self._streams.values())

    @property
    def stream_names(self) -> List[str]: return list(self._streams.keys())

    @property
    def delay(self) -> int: return int(1000 / min(stream.metadata.fps for stream in self._streams.values()))

    @property
    def num_frames(self) -> int: return self._num_frames

    def play(
        self,
        start         : int                                                   = 0,
        end           : int | None                                            = None, 
        skip_frames   : int                                                   = 1,
        window_size   : Size2D | Dict[str, Dict[str, Size2D] | Size2D] | None = None,
        delay         : int | None                                            = None,
        exclude_views : Dict[str, List[str]] | None                           = None
    ):
        ''' 
        Stream the videos from start to end frame in a synchronized manner.
        Each video will be displayed in its own window.
        '''

        def stream_view_name(stream: str, view: str) -> str: return f'{stream} - {view}'

        def parse_window_size(window_size: Size2D | Dict[str, Dict[str, Size2D] | Size2D] | None) -> Dict[str, Size2D]:

            window_size_ : Size2D | Dict[str, Dict[str, Size2D] | Size2D] = default(window_size, dict())

            window_size__ : Dict[str, Dict[str, Size2D] | Size2D]
            window_size__ = window_size_ if isinstance(window_size_, dict) else {name: window_size_ for name in self.stream_names}

            window_size_dict_out = dict()

            for stream_name, stream in self._streams.items():

                default_size = self._streams[stream_name].metadata.size

                stream_views = window_size__.get(stream_name, default_size)
                stream_views_ = stream_views if isinstance(stream_views, dict) else {view: stream_views for view in stream.views}

                for view in stream.views:
                    window_size_dict_out[stream_view_name(stream=stream_name, view=view)] = stream_views_.get(view, default_size)

            return window_size_dict_out
        
        def parse_active_views(exclude_views: Dict[str, List[str]] | None) -> Dict[str, bool]:

            exclude_views_ : Dict[str, List[str]] = default(exclude_views, dict())
            
            return {
                stream_view_name(stream=stream_name, view=view): view not in exclude_views_.get(stream_name, [])
                for stream_name, stream in self._streams.items()
                for view       in stream.views
            }

        delay_ : int = default(delay, self.delay)
        end_   : int   = default(end,   self._num_frames)
        
        windows_size_ : Dict[str, Size2D] = parse_window_size(window_size)
        active_views  : Dict[str, bool]   = parse_active_views(exclude_views)

        assert windows_size_.keys() == active_views.keys(), "Window size and active views must have the same keys."
        assert len(windows_size_) <= len(self.VIEW_KEYS), "Too many views to display."

        view_keys = {
            key : view
            for key, view in zip(self.VIEW_KEYS[:len(windows_size_)], windows_size_.keys())
        }

        # print('Debug')
        # print(windows_size_)
        # print(active_views)
        # print(view_keys)
        # return

        for stream_name, stream in self._streams.items():

            for view in stream.views:

                name       = stream_view_name(stream=stream_name, view=view)
                view_size  = windows_size_[name]
                active     = active_views [name]

                if active:
                    cv.namedWindow (name, cv.WINDOW_NORMAL)
                    cv.resizeWindow(name, *view_size)

        # Create iterators for each video stream
        iterators = [
            stream.iter_range(start=start, end=end_, step=skip_frames) 
            for _, stream in self._streams.items()
        ]

        try:

            self._logger.info(msg=f'Stream started. Press {self.EXIT_KEY} to exit.')
            self._logger.info(msg=f'Press key to activate or deactivate views:')
            for k, n in view_keys.items(): self._logger.info(msg=f'  {k} - {n}')
            self._logger.info(msg=f'')

            # Playback loop
            while True:

                # Read frames from all iterators
                for stream, iterator in zip(self._streams.values(), iterators):
                    try:
                        frame_views = next(iterator) # type: ignore
                        for view, frame in frame_views.items():
                            name = stream_view_name(stream=stream.name, view=view)
                            if active_views[name]:  cv.imshow(name, cv.cvtColor(frame, cv.COLOR_RGB2BGR))
                    except StopIteration:
                        # If any stream is finished, exit playback
                        cv.destroyAllWindows()
                        return

                key = chr(cv.waitKey(int(delay_)) & 0xFF)
                
                if key == self.EXIT_KEY:
                    self._logger.info(msg="Exiting video playback.")
                    break

                if key in view_keys:

                    name = view_keys[key]
                    
                    if active_views[name]:

                        # Close the window
                        cv.destroyWindow(view_keys[key])
                        active_views[view_keys[key]] = False
                    
                    else:

                        # Reopen the window
                        cv.namedWindow(name, cv.WINDOW_NORMAL)
                        cv.resizeWindow(name, *windows_size_[name])
                        active_views[name] = True
                    
                    if all(not active for active in active_views.values()):
                        self._logger.info(msg="All views are disabled, exiting video playback.")
                        break

            # Close all windows
            cv.destroyAllWindows()
        
        except Exception as e:
            self._logger.error(msg=f"Error occurred during video playback: {e}")
            cv.destroyAllWindows()
            raise e
