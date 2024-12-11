from __future__ import annotations

from abc import abstractmethod
import logging
from logging import Logger
import os
from typing import Callable, List, Set, Tuple
from loguru import logger as loguru_logger

# --- LOGGER ---

class FormatLogger(logging.Logger):

    def __init__(self, name: str): 
        
        super().__init__(name=name)

        self._formatter = lambda x: x

    @property
    def formatter(self) -> Callable[[str], str]: return self._formatter
    
    @formatter.setter
    def formatter(self, prefix: Callable[[str], str]): self._formatter = prefix
    
    def reset_formatter(self): self._formatter = lambda x: x

    def info   (self, msg): self._info   (self.formatter(msg))
    def warning(self, msg): self._warning(self.formatter(msg))
    def error  (self, msg): self._error  (self.formatter(msg))

    @abstractmethod
    def _info   (self, msg): raise NotImplementedError

    @abstractmethod
    def _warning(self, msg): raise NotImplementedError

    @abstractmethod
    def _error  (self, msg): raise NotImplementedError

class PrintLogger(FormatLogger):

    def __init__(self): super().__init__(name='PrintLogger')

    def _info   (self, msg): print(f"INFO:  {msg}")
    def _warning(self, msg): print(f"WARN:  {msg}")
    def _error  (self, msg): print(f"ERROR: {msg}")

class SilentLogger(FormatLogger):

    def __init__(self): super().__init__(name='SilentLogger')

    def _info   (self, msg): pass
    def _warning(self, msg): pass
    def _error  (self, msg): pass

class FileLogger(FormatLogger):

    def __init__(self, file, level=logging.INFO):

        super().__init__(name='FileLogger')

        InputSanitization.check_extension(path=file, ext='.log')

        loguru_logger.add(file, level=level)
        self._logger = loguru_logger

    def _info   (self, msg): self._logger.info   (msg)
    def _warning(self, msg): self._logger.warning(msg)
    def _error  (self, msg): self._logger.error  (msg)


# --- IO OPERATIONS ---

class PathOperations:

    @staticmethod
    def get_file(path: str) -> str:
        ''' Get the file name from the path '''

        return os.path.basename(path)
    
    @staticmethod
    def get_file_name(path: str) -> str:
        ''' Get the file name from the path '''

        file, ext = os.path.splitext(PathOperations.get_file(path))
        return file
    
    @staticmethod
    def get_file_extension(path: str) -> str:
        ''' Get the file extension from the path '''

        file, ext = os.path.splitext(path)
        return ext[1:]
    
    @staticmethod
    def get_containing_folder(path: str) -> str:
        ''' Get the containing folder of the file '''

        return os.path.basename(os.path.dirname(path))

class IOOperations:

    @staticmethod
    def make_dir(path: str, logger: FormatLogger = SilentLogger()):
        ''' Create a directory if it does not exist '''

        if not os.path.exists(path): 
            os.makedirs(path)
            logger.info(f"Directory created at: {path}")
        else:
            logger.info(f"Directory found at: {path}")


class InputSanitization:

    @staticmethod
    def check_input(path: str, logger: FormatLogger = SilentLogger()):
        ''' Check if input file exists '''

        if not os.path.exists(path): 
            raise FileNotFoundError(f"Input file not found: {path}")
        else:
            logger.info(f"Input file found at: {path} ")

    @staticmethod
    def check_output(path: str, logger: FormatLogger = SilentLogger()):
        ''' Check if the directory of the output file exists '''

        out_dir = os.path.dirname(path)

        if not os.path.exists(out_dir): 
            raise FileNotFoundError(f"Output directory not found: {out_dir}")
        else:                           
            logger.info(f"Output directory found at: {out_dir} ")

    @staticmethod
    def check_extension(path: str, ext: str | Set[str]):
        ''' Check if any of the extensions in the list match the file extension '''

        if type(ext) == str: ext = {ext}

        if not any([path.endswith(e) for e in ext]): 
            raise ValueError(f"Invalid file extension: {path}. Expected one of {ext} extensions.")

