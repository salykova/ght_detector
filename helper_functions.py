#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Multiple helper functions for visualizing results

"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def transform_points(points: np.ndarray, dx: float, dy: float, scaling: float, rotation: float) -> np.ndarray:
    """Transform the points according to the given translation, rotation and scale.

    Transform multiple points given as np.ndarray with shape (n, 2) according to the given translation, rotation, scale.
    We will use this function to transform the vertices (corners) of a rectangle representing our detected object.
    The rectangle is defined via the 4 vertices [[0, 0], [width, 0], [width, height], [0, height]] with width and height
    being the dimensions of the object image. These are therefore the positions of the corners in the object coordinate
    frame. Use homogeneous coordinates to be able to vectorize the transformation.
    Using this function, we can transform the rectangle into the image plane according to the keypoint matches.
    You can check the function draw_rectangles to see how transform_points is used.
    For more information about transformations refer to Szeliski Book 2.1.2: 2D transformations

    :param points: Points to be transformed. Each row holds the (x, y) coordinates in the origin coordinate frame
    :type points: np.ndarray with shape (n, 2)

    :param dx: Translation in x direction of the origin of the object coordinate frame in the scene coordinate frame
    :type dx: float

    :param dy: Translation in y direction of the origin of the object coordinate frame in the scene coordinate frame
    :type dy: float

    :param scaling: Scaling factor
    :type scaling: float

    :param rotation: Rotation in clockwise rotation in rad
    :type rotation: float

    :return: Transformed points in the target coordinate frame (scene coordinate frame)
    :rtype: np.ndarray with shape (n, 2)
    """
    ######################################################
    # Write your own code here
    transformed_points = []
    R = np.array([[np.cos(rotation), -np.sin(rotation), 0], [np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    T = np.array([[scaling, 0, dx], [0, scaling, dy]])
    for point in points:
        transformed_points.append(np.matmul(T, np.matmul(R, np.r_[point,1])))
    transformed_points = np.array(transformed_points)
    ######################################################
    return transformed_points


def draw_rectangles(scene_img: np.ndarray,
                    object_img: np.ndarray,
                    object_configurations: np.ndarray) -> np.ndarray:
    """Plot rectangles with size of object_img into scene_img given the transformation parameters x,y,s,a

    :param scene_img: Image to draw rectangles into
    :type scene_img: np.ndarray with shape (height, width, channels)

    :param object_img: Image of the searched object which defines the size of the rectangles before transformation
    :type object_img: np.ndarray with shape (height, width, channels)

    :param object_configurations: Array with shape (n, 4), each row holding one detected object configuration (x,y,s,o)
        x, y: coordinates of top-left corner of detected object in the scene image coordinate frame
        s: relative scale of the object between object coordinate frame and scene image coordinate frame
        o: orientation (clockwise)
    :type object_configurations: np.array with shape (n, 4) with n being the number of detected objects

    :return: Copied image of scene_img with rectangles drawn on top
    :rtype: np.ndarray with  the same shape (height, width, channels) as scene_img
    """
    output_img = scene_img.copy()

    # Get the height and width of our template object which will define the size of the rectangles we draw
    height, width = object_img.shape[0:2]

    # Define a rectangle with the 4 vertices. With the top left vertex at position [0,0]
    rectangle = np.array([[0, 0],
                          [width, 0],
                          [width, height],
                          [0, height]], dtype=np.float32)

    # Iterate over all found object configurations, transform the rectangles to the scene image coordinate frame
    # according to the configuration, and draw them using cv2.polylines
    for conf in object_configurations:
        rectangle_tf = np.around(transform_points(rectangle, conf[0], conf[1], conf[2], conf[3])).astype(np.int32)
        cv2.polylines(output_img, [rectangle_tf], isClosed=True, color=(0, 255, 0), thickness=3)

        # Change the top line to be blue, so we can tell the top of the object
        cv2.line(output_img, tuple(rectangle_tf[0]), tuple(rectangle_tf[1]), color=(255, 0, 0), thickness=3)

    return output_img


def show_image(img: np.ndarray, title: str, save_image: bool = False, use_matplotlib: bool = False) -> None:
    """ Plot an image with either OpenCV or Matplotlib.

    :param img: :param img: Input image
    :type img: np.ndarray with shape (height, width) or (height, width, channels)

    :param title: The title of the plot which is also used as a filename if save_image is chosen
    :type title: string

    :param save_image: If this is set to True, an image will be saved to disc as title.png
    :type save_image: bool

    :param use_matplotlib: If this is set to True, Matplotlib will be used for plotting, OpenCV otherwise
    :type use_matplotlib: bool
    """

    # First check if img is color or grayscale. Raise an exception on a wrong type.
    if len(img.shape) == 3:
        is_color = True
    elif len(img.shape) == 2:
        is_color = False
    else:
        raise ValueError(
            'The image does not have a valid shape. Expected either (height, width) or (height, width, channels)')

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.

    elif img.dtype == np.float64:
        img = img.astype(np.float32)

    if use_matplotlib:
        plt.figure()
        plt.title(title)
        if is_color:
            # OpenCV uses BGR order while Matplotlib uses RGB. Reverse the the channels to plot the correct colors
            plt.imshow(img[..., ::-1])
        else:
            plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        cv2.imshow(title, img)
        cv2.waitKey(0)

    if save_image:
        if is_color:
            png_img = (cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) * 255.).astype(np.uint8)
        else:
            png_img = (cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA) * 255.).astype(np.uint8)
        cv2.imwrite(title.replace(" ", "_") + ".png", png_img)
