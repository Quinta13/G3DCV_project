# **Geometric and 3D Computer Vision - a.y. 2024/25**  
**Sebastiano Quintavalle**  

## **Overview**  

This project involves the implementation of the **Reflectance Transformation Imaging (RTI)** technique as described in the [paper](https://www.researchgate.net/publication/368520914_On-the-Go_Reflectance_Transformation_Imaging_with_Ordinary_Smartphones) *On-the-go Reflectance Transformation Imaging with Ordinary Smartphones* by M. Pistellato and F. Bergamasco (2022).  

The code is written in **Python** and is organized into three main directories:  
- **`src`**: Contains the source code, structured into modules, and includes the main classes implementing the RTI technique.  
- **`scripts`**: Contains five executable scripts that implement the necessary preprocessing steps for the RTI technique.  
- **`notebooks`**: Includes Jupyter notebooks that provide interactive demonstrations of specific components, particularly to visualize the impact of hyperparameters.  

The `src` directory is further divided into two main modules:  
- **`model`**: Defines the core classes implementing the RTI technique.  
- **`utils`**: Contains helper functions for I/O operations, settings management, and visualization.  

---

## **Source Code**  

The **`src/utils`** directory contains the following files:  

- **`settings.py`** - Defines key macros and parameters parsed from `.env` configuration files.  
- **`io_.py`** - Provides core I/O functions for handling file paths, logging, and processing audio/video files.  
- **`stream.py`** - Defines classes for video streaming, enabling concurrent visualization of multiple processing steps that can be toggled on or off.  
- **`calibration.py`** - Defines a data class to store camera calibration settings, which can be saved and loaded from a file.  
- **`misc.py`** - Includes general-purpose utility functions.  

The **`src/model`** directory contains the following files:  

- **`typing.py`** - Defines type aliases used in other modules.  
- **`geom.py`** - Implements geometric concepts such as points, contours, and vectors representing light direction for frame manipulation.  
- **`preprocessing.py`** - Implements preprocessing operations required before applying the RTI technique, including (1) video synchronization and frame rate alignment, and (2) camera calibration.  
- **`thresholding.py`** - Implements image binarization techniques for marker detection.  
- **`marker.py`** - Contains logic for marker detection in videos, along with operations such as drawing markers on frames, warping objects to their center, and estimating camera pose.  
- **`mlic.py`** - Defines the **Multi-Light Image Collection (MLIC)** class, which facilitates dataset manipulation and file storage. It also includes classes for collecting MLIC from synchronized video streams by warping the static camera frame and estimating the dynamic camera position.  
- **`interpolation.py`** - Implements interpolation techniques for each pixel in the MLIC and constructs basis functions fitted to the collected data. It also provides a class for collecting basis functions per pixel, reconstructing the image, and saving the result. The supported interpolation methods include **Radial Basis Functions (RBF)** and **Polynomial Texture Maps (PTM)**.  
- **`rti.py`** - Defines an **interactive RTI class**, allowing users to control the light source input and visualize the corresponding illuminated image in real time.  

---

## **Scripts**  

The scripts in the **`scripts`** directory are **configurable via the `.env` file** and require specific Python dependencies listed in `requirements.txt`. The available scripts are:  

- **`1_sync.py`** - Synchronizes the two video streams and aligns their frame rates. (Implements `src/model/preprocessing.py`).  
- **`2_calibrate.py`** - Calibrates the camera using a chessboard pattern video and saves the calibration parameters. (Implements `src/model/preprocessing.py`, `src/model/calibration.py`).  
- **`3_collect_mlic.py`** - Performs marker detection in the synchronized video streams to construct the **Multi-Light Image Collection (MLIC)**. (Implements `src/model/marker.py`, `src/model/mlic.py`, `src/model/geom.py`).  
- **`4_interpolate.py`** - Performs pixel-wise interpolation on the MLIC dataset and saves the basis functions. (Implements `src/model/interpolation.py`).  
- **`5_rti.py`** - Executes the **interactive RTI visualization**. (Implements `src/model/rti.py`).  

---

## **Notebooks**  

The **Jupyter notebooks** in the **`notebooks`** directory provide an interactive visualization of how different **hyperparameters** affect the RTI technique and justify specific implementation choices. The available notebooks are:  

- **`1_calibrated_camera.ipynb`** - Displays the camera calibration setup and the effects of undistortion.  
- **`2_thresholding.ipynb`** - Compares different binarization techniques for marker detection and their hyperparameters.  
- **`3_marker_detection.ipynb`** - Shows how hyperparameters influence marker detection accuracy.  
- **`4_mlic.ipynb`** - Visualizes the collected **Multi-Light Image Collection (MLIC)** dataset and examines the effect of various hyperparameters, particularly comparing geometric vs algebraic camera pose estimation methods.  
- **`5_interpolation.ipynb`** - Demonstrates interpolation techniques and their hyperparameters.  

---
