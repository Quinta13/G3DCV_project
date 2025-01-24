from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, Iterator, List, Tuple

import cv2 as cv
import numpy as np

from src.utils.typing import Frame, Size2D, Views
from src.utils.io_ import (
    SilentLogger, BaseLogger,
    PathUtils, InputSanitizationUtils as ISUtils,
    VideoFile
)
from src.utils.typing import default

class Stream(ABC):
    '''
    Abstract class representing a stream of frames. 
    A stream can be either a video file stored in the disk or generated on-demand computing frames online.
    A stream allows for multiple views of the stream content (for example showing different steps of the video processing) 
        that can be displayed simultaneously in different windows.
    '''

    def __init__(self, name: str, logger: BaseLogger = SilentLogger()):
        ''' The stream is initialized with a name to identify it and a logger to log messages. '''

        self._name           : str        = name
        self._logger         : BaseLogger = logger

    # --- PROPERTIES ---
    
    @property
    def name(self) -> str: return self._name

    @name.setter
    def name(self, value: str) -> None: self._name = value

    @property
    def delay(self) -> int: return 1
    ''' Delay between frames in milliseconds. '''

    @property
    def views(self) -> List[str]: return []
    ''' List of views available in the stream. '''

    # --- MAGIC METHODS ---

    def __str__(self)  -> str: return f'{self.__class__.__name__}[{self.name}; frames: {len(self)}]'
    def __repr__(self) -> str: return str(self)

    def __iter__(self) -> Iterator[Tuple[int, Views]]: return self.iter_range(start=0, end=len(self), step=1)
    ''' Iterate over all couples (frame-id, frame views) in the stream. '''

    # --- ABSTRACT METHODS ---

    @abstractmethod
    def __len__(self) -> int: pass
    ''' Return the number of frames in the stream. '''

    @abstractmethod
    def iter_range(self, start: int, end: int, step: int = 1) -> Iterator[Tuple[int, Views]]: pass
    ''' 
    Logic to iterate over a specific range of frame views in the stream. 
    The iterator returns a tuple with the frame index and the views of the frame.
    '''

    @abstractmethod
    def __getitem__(self, idx: int) -> Views: pass
    ''' Get a specific frame from the stream. '''

    @abstractmethod
    def _default_window_size(self) -> Size2D: pass
    ''' Default window size for the stream. '''

    # --- STREAM PLAY ---

    def play(
        self, 
        start        : int                               = 0,
        end          : int                        | None = None, 
        skip_frames  : int                               = 1,
        window_size  : Dict[str, Size2D] | Size2D | None = None,
        exclude_views: List[str]                  | None = None,
        delay        : int                        | None = None
    ):
        ''' 
        Play the video stream specifying the start and end frame, the number of frames to skip.
        It also allows to specify the window size for each view, the views to exclude from the stream and the delay between frames.

        :param start:        Index of the first frame to display, defaults to the first frame of the stream.
        :param end:          Index of the last frame to display, if None the last frame of the stream is used.
        :param skip_frames:  Number of frames to skip between each frame displayed, defaults to 1 meaning no frames are skipped.
        :param window_size:  Size of the window to display the stream, it can be
            - A dictionary with the size for each view. 
            - A single size for all views.
            Defaults to None, meaning the default size of the stream for all views is used.
        :param exclude_views: List of views to exclude from the stream display, defaults to None meaning all views are displayed.
        '''

        # Defaults
        delay_         = default(delay, self.delay)
        exclude_views_ = default(exclude_views, [])
        window_size_   = default(window_size, {})
        
        # NOTE: The actual streaming logic is implemented in the `SynchronizedVideoStream` class that handles multiple streams.
        #       This method uses its play method providing a single stream to display.

        single_sync_stream = SynchronizedVideoStream(streams=[self], logger=self._logger)

        single_sync_stream.play(
            start=start,
            end=end,
            skip_frames=skip_frames,
            window_size={self.name: window_size_},
            exclude_views={self.name: exclude_views_},
            delay=delay_
        )


class VideoStream(Stream):
    '''
    Class for streaming video files stored in the disk.
    '''

    def __init__(
        self, 
        path    : str,
        name    : str | None = None,
        logger  : BaseLogger = SilentLogger()
    ):
        '''
        Initialize a video stream from a video file stored in the disk. 
        It opens the video capture and reads the metadata of the video file.
        '''
        
        # File name, defaults to folder and file name
        name = default(name, PathUtils.get_folder_and_file(path=path))

        super().__init__(name=name, logger=logger)
        
        # Video path and metadata
        ISUtils.check_input    (path=path, logger=self._logger)
        ISUtils.check_extension(path=path, logger=self._logger, ext=VideoFile.VIDEO_EXT)

        self._path: str = path
        self._metadata: VideoFile.VideoMetadata = VideoFile.VideoMetadata.from_video_path(path=self.path, logger=self._logger)
        
        # Open the video stream
        self._video_capture : cv.VideoCapture = cv.VideoCapture(self.path)
        if not self._video_capture.isOpened():
            self._logger.handle_error(msg=f"Failed to open video stream at {self.path}", exception=RuntimeError)
    
    # --- MAGIC METHODS ---
    
    def __str__ (self) -> str: return f"{self.__class__.__name__}[{'; '.join(f'{k}: {v}' for k, v in self._str_params.items())}]"
    def __repr__(self) -> str: return str(self)

    @property
    def _str_params(self) -> Dict[str, Any]:
        ''' Utility function for string representation.'''

        w, h = self.metadata.size

        return {
            'name'   : self.name,
            'frames' : len(self),
            'size'   : f'{w}x{h} pixels',
            'views'  : len(self.views)
        }

    def __len__(self) -> int: return self.metadata.frames
    ''' Return the number of frames in the video. '''
        
    def __getitem__(self, idx: int) -> Views:
        ''' Get a specific frame view in the video. '''

        # Set the frame position and read the frame
        self._video_capture.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._video_capture.read()

        if ret: return self._process_frame(frame=frame, frame_id=idx)
        
        self._logger.handle_error(msg=f"[{self.name}]: Unable to read frame at index {idx}.", exception=IndexError)
    
    # --- PROPERTIES ---

    @property    
    def path(self) -> str: return self._path

    @property
    def metadata(self) -> VideoFile.VideoMetadata: return self._metadata
    
    @property
    def delay(self) -> int: return int(1000 / self.metadata.fps)
    ''' Compute delay in milliseconds using the video frame rate. '''

    @property
    def _default_window_size(self) -> Size2D: return self.metadata.size
    ''' Default window size is the same as the video size. '''

    # --- STREAMING ---

    def iter_range(self, start: int, end: int, step: int = 1) -> Iterator[Tuple[int, Views]]:
        ''' Return an iterator over a specific range of frames in the video. '''
        
        # Set the video position to the starting frame
        self._video_capture.set(cv.CAP_PROP_POS_FRAMES, start)

        frame_id = start

        # Iterate over the frames
        while frame_id < end and self._video_capture.isOpened():
        
            # Skip frames according to the step parameter
            for _ in range(step-1): 
                self._video_capture.grab()  # NOTE: This is faster than reading the frame
                frame_id += 1

            # Read the frame
            ret, frame = self._video_capture.read()
            
            # Check if the frame was read successfully
            # NOTE: This will cause a StopIteration exception when the end of the video is reached.
            if not ret: break

            # NOTE: OpenCV reads frames in BGR format, we convert it to RGB after reading.
            # Cast frame to RGB
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Process the frame
            views_ = self._process_frame(frame=frame, frame_id=frame_id)

            # Yield the frame views (implementing the iterator protocol)
            yield frame_id, views_

            # Move to the next frame
            frame_id += 1

    # --- DEBUGGING ---

    # NOTE: The `_process_frame` function may be called to apply some processing external to the stream (e.g. get the name of stream views).
    #       For this reason a specific frame_id is used as a debug frame-id to differentiate it from the actual frame ids.

    @property
    def _debug_frame(self) -> Frame: return np.zeros((*self.metadata.size, 3), dtype=np.uint8)
    ''' Debug frame used for testing or debugging as an empty frame. '''

    @property
    def _debug_frame_id(self) -> int: return -1
    ''' Frame id to use for debugging. '''

    def _is_debug(self, frame_id: int) -> bool: return frame_id < 0
    ''' Method to implement the debug logic in frame processing. '''

    # --- STREAM and VIEWS ---

    @property
    def views(self) -> List[str]:  
        '''
        List of views available in the video stream.
        NOTE: This method uses a debug frame to get the available views.
        '''
        
        return list(self._process_frame(frame=self._debug_frame, frame_id=self._debug_frame_id).keys())

    def _process_frame(self, frame: Frame, frame_id: int) -> Views:
        '''
        Process a frame from the video stream.
        In this simple video stream, a single view of the original frame is returned.

        NOTE: This method can be overridden by subclasses to apply custom processing.
        '''

        return {'raw': frame}


class SynchronizedVideoStream:
    '''
    Class to synchronize multiple video streams and display them simultaneously.
    '''
    
    EXIT_KEY  = 'q'
    ''' Key to exit the video playback. '''

    VIEW_KEYS = '1234567890abcdefghijklmnoprstuvwxyz'
    ''' Keys to activate or deactivate views in the video playback. '''

    def __init__(
        self,
        streams: List[Stream],
        logger: BaseLogger = SilentLogger()
    ):
        ''' Initialize a synchronized video stream object from multiple video streams. '''

        self._logger        : BaseLogger = logger

        # A) Check at least one video stream
        if not streams:
            self._logger.handle_error(msg="At least one video stream must be provided.", exception=ValueError)

        # B) Check all different names, as they will be indexed by name which must be unique
        repeated_names = {k for k, c in Counter([s.name for s in streams]).items() if c > 1}
        if repeated_names:
            self._logger.handle_error(
                msg=f"Video stream names must be unique. Got repeated names: {repeated_names}",
                exception=ValueError
            )
        
        # C) Check if all streams have the same number of frames, in case of different lengths, warn the longest ones will be truncated
        
        # Get the number of frames in each stream
        stream_lengths = {len(stream) for stream in streams}  
        
        # Use the minimum number of frames for the video playback
        self._num_frames: int = min(stream_lengths)    

        # Warn if streams have different number of frames
        if len(stream_lengths) > 1:
            self._logger.warning(msg=f'Streams have different number of frames. Using the minimum number of frames: {self._num_frames}.')
            for stream in streams:
                truncated = len(stream) - self._num_frames
                if truncated > 0:                    
                    self._logger.warning(
                        msg=f" > Stream {stream.name} has {len(stream)} frames, truncating last {truncated} frame{'s' if truncated > 1 else ''}."
                    )
            self._logger.info('')

        # Save streams indexed by name
        self._streams: Dict[str, Stream] = {stream.name: stream for stream in streams}

    # --- MAGIC METHODS ---
    
    def __str__(self) -> str: 
        return f"SynchronizedVideoStream[{len(self)} streams ({', '.join(self._streams.keys())}); frames: {self._num_frames}]"
    
    def __repr__(self) -> str: return str(self)

    def __len__(self) -> int: return len(self._streams)
    ''' Return the number of video streams in the synchronized stream. '''

    # --- PROPERTIES --- 
    
    @property
    def stream_names(self) -> List[str]: return list(self._streams.keys())

    @property
    def streams(self) -> List[Stream]: return list(self._streams.values())

    @property
    def num_frames(self) -> int: return self._num_frames

    @property
    def delay(self) -> int: return max(stream.delay for stream in self._streams.values())
    ''' Use the maximum delay between all streams. '''
    
    # --- STREAM PLAY ---

    def _play_iter_progress(self, views: Dict[str, Dict[str, Frame]], frame_id: int): pass
    ''' 
    Method to implement the progress logic during video playback. 
    Default implementation is a no-op.

    :param views: Dictionary of processed views for every stream at the current frame_id.
    :param frame_id: Index of the current frame being displayed.
    '''


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
        Play each stream in a synchronized manner. 
        Each video view will be displayed in its own window, and a key can be pressed to activate or deactivate each view.
        It allows to control the start and end frame of the video playback and the number of frames to skip.
        It also provides some utilities to control the window size of each view, the delay between frames and the views to exclude.

        :param start:        Index of the first frame to display, defaults to the first frame of the stream.
        :param end:          Index of the last frame to display, if None the last frame of the stream is used.
        :param skip_frames:  Number of frames to skip between each frame displayed, defaults to 1 meaning no frames are skipped.
        :param window_size:  Size of the window to display the stream, it can be
            - A unique frame size for all streams and views.
            - A dictionary specifying the size for each stream.
            - A nested dictionary specifying the size for each stream and view.
            When there's no specification either for a stream or for a view, the default size of the stream is used.
        :param exclude_views: Dictionary specifying the views to exclude from each of the displayed stream.
        '''

        def stream_view_name(stream: str, view: str) -> str: return f'{stream} - {view}'
        ''' String to index the (stream, view) pair. '''

        def parse_window_size(window_size: Size2D | Dict[str, Dict[str, Size2D] | Size2D] | None) -> Dict[str, Size2D]:
            ''' Parse the window size parameter to get the size for each view. '''

            window_size_ : Size2D | Dict[str, Dict[str, Size2D] | Size2D] = default(window_size, dict())
            window_size__: Dict[str, Dict[str, Size2D] | Size2D] 
            window_size__ = window_size_ if isinstance(window_size_, dict) else {name: window_size_ for name in self.stream_names}

            window_size_dict_out = dict()

            for stream_name, stream in self._streams.items():

                default_size  = self._streams[stream_name]._default_window_size
                stream_views  = window_size__.get(stream_name, default_size)
                stream_views_ = stream_views if isinstance(stream_views, dict) else {view: stream_views for view in stream.views}

                for view in stream.views:
                    window_size_dict_out[stream_view_name(stream=stream_name, view=view)] = stream_views_.get(view, default_size)

            return window_size_dict_out
        
        def parse_active_views(exclude_views: Dict[str, List[str]] | None) -> Dict[str, bool]:
            ''' Parse the exclude views parameter to get the active views for each stream. '''

            exclude_views_ : Dict[str, List[str]] = default(exclude_views, dict())
            
            return {
                stream_view_name(stream=stream_name, view=view): view not in exclude_views_.get(stream_name, [])
                for stream_name, stream in self._streams.items()
                for view       in stream.views
            }

        # Default values
        delay_ : int = default(delay, self.delay)
        end_   : int = default(end,   self._num_frames)

        # Check input
        if start < 0:
            self._logger.handle_error(msg=f"Start frame {start} is less than 0.", exception=ValueError)

        if start > end_:
            self._logger.handle_error(msg=f"Start frame {start} is greater than end frame {end}.", exception=ValueError)

        if end_ > self._num_frames: 
            self._logger.handle_error(msg=f"End frame {end_} is greater than the number of frames {self._num_frames}.", exception=ValueError)
        
        # Parse window size and active views
        windows_size_ : Dict[str, Size2D] = parse_window_size(window_size)
        active_views  : Dict[str, bool]   = parse_active_views(exclude_views)

        assert windows_size_.keys() == active_views.keys(), "Window size and active views must have the same keys."
        assert len(windows_size_) <= len(self.VIEW_KEYS),   "Too many views to display."

        # Assign a key to each stream and view to activate or deactivate it
        view_keys = {
            key : view
            for key, view in zip(self.VIEW_KEYS[:len(windows_size_)], windows_size_.keys())
        }

        # Create a window for each active view and set the size 
        for stream_name, stream in self._streams.items():

            for view_name in stream.views:

                name       = stream_view_name(stream=stream_name, view=view_name)
                view_size  = windows_size_[name]
                active     = active_views [name]

                if active:
                    cv.namedWindow (name, cv.WINDOW_NORMAL)
                    cv.resizeWindow(name, *view_size)

        # Get the iterators for each video stream
        iterators = [
            stream.iter_range(start=start, end=end_, step=skip_frames) 
            for stream in self._streams.values()
        ]

        # The playing logic is implemented in a try-catch block to handle any possible exceptions and close all windows
        try:

            # Logging preamble
            self._logger.info(msg=f'Stream started. Press {self.EXIT_KEY} to exit.')
            self._logger.info(msg=f'Press key to activate or deactivate views:')
            for k, n in view_keys.items(): self._logger.info(msg=f'  {k} - {n}')
            self._logger.info(msg=f'')

            # Playback loop
            while True:
                
                # Save the processed views for every stream
                curr_frame_views = dict()

                for stream, iterator in zip(self._streams.values(), iterators):

                    try:

                        # Save the processed views for the current stream
                        progress_view = dict()

                        # Get the next frame views
                        frame_id, views = next(iterator)

                        # Iterate over the views and display them if active
                        for view_name, frame in views.items():

                            name = stream_view_name(stream=stream.name, view=view_name)

                            # Display the view if active
                            if active_views[name]:  
                                frame_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # NOTE: OpenCV displays images in BGR format
                                cv.imshow(name, frame_bgr)
                            
                            progress_view[view_name] = frame
                        
                        curr_frame_views[stream.name] = progress_view
                    
                    # If any stream is finished, exit playback
                    except StopIteration:
                        cv.destroyAllWindows()
                        return
                
                # Playback progress with the current frame views
                self._play_iter_progress(views=curr_frame_views, frame_id=frame_id)

                # Check for key press for exiting or activating/deactivating views
                key = chr(cv.waitKey(delay_) & 0xFF)
                
                # Exiting
                if key == self.EXIT_KEY:
                    self._logger.info(msg="Exiting video playback.")
                    break

                # Activate or deactivate views
                if key in view_keys:

                    name = view_keys[key]
                    
                    # If it was active, we close the window
                    if active_views[name]:
                        cv.destroyWindow(view_keys[key])
                        active_views[view_keys[key]] = False
                    
                    # Otherwise, we open the window
                    else:
                        cv.namedWindow(name, cv.WINDOW_NORMAL)
                        cv.resizeWindow(name, *windows_size_[name])
                        active_views[name] = True
                    
                    # In the case all views are disabled, we exit the video playback
                    if all(not active for active in active_views.values()):
                        self._logger.info(msg="All views are disabled, exiting video playback.")
                        break

            # Close all windows
            cv.destroyAllWindows()
        
        # Handle any exception occurred during video playback
        except Exception as e:
            cv.destroyAllWindows()
            self._logger.handle_error(msg=f"Error occurred during video playback: {e}", exception=type(e))
