# Contour Detection and Inflection Point Identification

This project provides a Python script for detecting contours and identifying inflection points within images, using OpenCV. It specifically focuses on finding areas of interest, calculating distances between contours, and marking inflection points based on specified threshold values.

## Features

- **Contour Detection**: Identifies and draws contours of objects within an image based on area, width, and height thresholds.
- **Distance Measurement**: Calculates and displays the distance between adjacent contours, considering only those with significant width.
- **Inflection Point Identification**: Finds and marks the 'M' (maximum) and 'W' (minimum) inflection points along the top and bottom boundaries of detected contours.

## Requirements

- Python 3.x
- OpenCV (`cv2`) library

## Installation

Ensure you have Python 3.x installed on your system. You can then install OpenCV using pip:

```bash
pip install opencv-python-headless


## Usage
Run the script by specifying the path to your target image as follows:

```bash
python process_image.py <image_path> <dist_th> <height_th>


## Example

```bash
python process_image.py test.bmp 50 3
