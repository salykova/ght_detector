#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Vision and Cognitive Robotics
Matthias Hirschmanner 2020
Automation & Control Institute, TU Wien
"""

from pathlib import Path

import numpy as np
import cv2

from detect_objects import detect_objects
from helper_functions import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # You can change the parameters here. You should not need to change anything else
    image_nr = 3
    save_image = False
    use_matplotlib = False
    debug_output = False  # <<< change to reduce output when you're done

    # Get path
    current_path = Path(__file__).parent

    # Load images.
    # scene_img  -> image in which we want to detect the object (trainImage in OpenCV)
    # object_img -> image of the object we want to detect (queryImage in OpenCV)
    scene_img = cv2.imread(str(current_path.joinpath("data/img")) + str(image_nr) + ".jpg")
    if scene_img is None:
        raise FileNotFoundError("Couldn't load image in " + str(current_path))
    scene_img_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)  # trainImage

    object_img = cv2.imread(str(current_path.joinpath("data/obj3.jpg")))
    if object_img is None:
        raise FileNotFoundError("Couldn't load image in " + str(current_path))
    object_img_gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)  # queryImage

    # Get SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    scene_keypoints, scene_descriptors = sift.detectAndCompute(scene_img_gray, None)
    object_keypoints, object_descriptors = sift.detectAndCompute(object_img_gray, None)

    # FLANN (Fast Library for Approximate Nearest Neighbors) parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # For each keypoint in object_img get the 3 best matches
    matches = flann.knnMatch(object_descriptors, scene_descriptors, k=120)

    if debug_output:
        # Only show every fifth match, otherwise it gets to overwhelming
        match_mask = np.zeros(np.array(matches).shape, dtype=np.int)
        match_mask[::5, ...] = 1

        draw_params = dict(matchesMask=match_mask,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        matches_img = cv2.drawMatchesKnn(object_img,
                                         object_keypoints,
                                         scene_img,
                                         scene_keypoints,
                                         matches,
                                         None,
                                         **draw_params)
        show_image(matches_img, "Matches",  save_image=save_image, use_matplotlib=use_matplotlib)

    # Detecting object configurations
    object_configurations = detect_objects(scene_img,
                                           object_img,
                                           scene_keypoints,
                                           object_keypoints,
                                           matches,
                                           debug_output=debug_output)
    # Plot results
    plot_img = draw_rectangles(scene_img, object_img, object_configurations)
    show_image(plot_img, "Final Result", save_image=save_image, use_matplotlib=use_matplotlib)
