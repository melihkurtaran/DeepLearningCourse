#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 11:42:10 2022

"""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt


def inspect_obj(obj, internal=False):
    """Return the attributes (properties and methods) of an object"""

    """Return a dictionary with three elements. The first element has
    'properties' as key and its value is a list of strings with all the
    properties that the dir() function was able to retrieve.
    The second element of the dictionary, with 'methods' key, is the equivalent
    applied to methods.
    The third element is the union of the previous two, and it's key is
    'attributes'.
    You might want to take a look at the 'inspect' library if you need to dig
    deeper. An example of use would be:
    print(inspect_obj(obj)['properties'])

    Parameters
    ----------
    obj :
        TYPE: object
        DESCRIPTION: It can be any object.
    internal :
        TYPE: bool
        DESCRIPTION: If True it also returns the attributes that start with
            underscore.

    Returns
    -------
    output :
        TYPE: Dictionary of two elements of the type list of strings.
        DESCRIPTION. Dictionary with two elements. The first is
            output['properties'] and the second is output['methods']. They list
            the properties and methods respectively.
    """

    dir_obj = []

    # Loop through attributes found by dir(). This first filter is done because
    # sometimes there are attributes that raise an error when called by
    # getattr() due to they haven't been initialized, or due to they have a
    # special behavior.
    for func in dir(obj):
        try:
            _ = getattr(obj, func)
        except:
            continue
        dir_obj.append(func)

    # Selection of methods and properties
    if internal:
        method_list = [func for func in dir_obj if callable(getattr(obj,
                                                                    func))]
        property_list = [prop for prop in dir_obj if prop not in method_list]
    else:
        method_list = [func for func in dir_obj if callable(
            getattr(obj, func)) and not func.startswith('_')]
        property_list = [prop for prop in dir_obj if
                     prop not in method_list and not prop.startswith('_')]

    return {'properties': property_list, 'methods': method_list,
        'attributes': sorted(property_list + method_list)}


# %%

def p_gen3(n, shuffle=True, seed=42):
    """Function to generate a cloud of points"""

    """
    This function is specific to this lab. They have no use nowhere else.
    """
    def points(n, xmin, xmax, ymin, ymax):
        x = xmin * np.ones(n, dtype=np.float32) +\
            (xmax - xmin) * np.random.random(n)
        y = ymin * np.ones(n, dtype=np.float32) +\
            (ymax - ymin) * np.random.random(n)
        return x, y

    n1 = math.trunc(0.33 * n)
    n2 = math.trunc(0.16 * n)
    n3 = math.trunc(0.17 * n)
    n4 = n - n1 - n2 - n3

    xmin, xmax = 0.4, 1.3
    ymin, ymax = 0.4, 1.4

    # 2 rectangular blocks of red dots
    np.random.seed(seed)
    pr1_x, pr1_y = points(n1, xmin, 1.0, ymin, 0.9)
    pr1 = np.c_[pr1_x, pr1_y]
    
    # Cradle
    pr1[0] = np.array([0.8, 1.0], dtype=np.float32)

    pr3_x, pr3_y = points(n3, xmin, 0.7, 0.9, ymax)
    pr3 = np.c_[pr3_x, pr3_y]

    t1 = np.r_[pr1, pr3]
    red_inputs = np.c_[t1, np.zeros((t1.shape[0], 1))]

    # 2 rectangular blocks of blue dots
    pr2_x, pr2_y = points(n2, 1.0, xmax, ymin, 0.9)
    pr2 = np.c_[pr2_x, pr2_y]

    # Cradle
    pr2[-1] = np.array([0.75, 0.8], dtype=np.float32)
    pr2[-2] = np.array([0.91, 0.5], dtype=np.float32)

    pr4_x, pr4_y = points(n4, 0.7, xmax, 0.9, ymax)
    pr4 = np.c_[pr4_x, pr4_y]

    t2 = np.r_[pr2, pr4]
    blue_inputs = np.c_[t2, np.ones((t2.shape[0], 1))]

    inputs = np.concatenate((red_inputs, blue_inputs), axis=0)
    if shuffle:
        np.random.shuffle(inputs)
    inputs = inputs.T

    tst = np.empty((3, 1), dtype=np.float32)

    # Loop through the image. Initially all of them red
    for x in np.linspace(xmin, xmax, 81):
        for y in np.linspace(ymin, ymax, 81):
            tst = np.append(tst, np.array([[x], [y], [0]],
                                          dtype=np.float32), axis=1)

    return tst[:, 1:].astype(np.float32), inputs.astype(np.float32)

# %%


def p_gen2(n, shuffle=True, seed=42):
    """Dedicated function to generate dots"""

    """
    This function is specific to this lab. They have no use nowhere else.
    """
    def points(n, xmin, xmax, ymin, ymax):
        x = xmin * np.ones(n, dtype=np.float32) +\
            (xmax - xmin) * np.random.random(n)
        y = ymin * np.ones(n, dtype=np.float32) +\
            (ymax - ymin) * np.random.random(n)
        return x, y

    n1 = math.trunc(0.26 * n)
    n2 = math.trunc(0.29 * n)
    n3 = math.trunc(0.08 * n)
    n4 = math.trunc(0.17 * n)

    nb = n - n1 - n2 - n3 - n4

    xmin, xmax = 0.4, 1.4
    ymin, ymax = 0.4, 1.0

    # 4 rectangular blocks of red dots
    np.random.seed(seed)
    pr1_x, pr1_y = points(n1, xmin, 0.85, ymin, 0.8)
    pr1 = np.c_[pr1_x, pr1_y]
    pr2_x, pr2_y = points(n2, xmin, 1.3, 0.78, ymax)
    pr2 = np.c_[pr2_x, pr2_y]
    pr3_x, pr3_y = points(n3, 0.8, xmax, ymin, 0.5)
    pr3 = np.c_[pr3_x, pr3_y]
    pr4_x, pr4_y = points(n4, 1.18, xmax, 0.48, ymax)
    pr4 = np.c_[pr4_x, pr4_y]

    t1 = np.r_[pr1, pr2, pr3, pr4]
    red_inputs = np.c_[t1, np.zeros((t1.shape[0], 1))]

    # 1 rectangular block of blue dots
    pb_x, pb_y = points(nb, 0.8, 1.2, 0.5, 0.85)
    t2 = np.c_[pb_x, pb_y]
    blue_inputs = np.c_[t2, np.ones((t2.shape[0], 1))]

    inputs = np.concatenate((red_inputs, blue_inputs), axis=0)
    if shuffle:
        np.random.shuffle(inputs)
    inputs = inputs.T

    tst = np.empty((3, 1), dtype=np.float32)

    # Loop through the image. Initially all of them red
    for x in np.linspace(xmin, xmax, 81):
        for y in np.linspace(ymin, ymax, 81):
            tst = np.append(tst, np.array([[x], [y], [0]],
                                          dtype=np.float32), axis=1)

    return tst[:, 1:].astype(np.float32), inputs.astype(np.float32)


# %%

def p_gen1(num_points, seed=43):
    """Dedicated function to generate dots"""

    """
    This function is specific to this lab. They have no use nowhere else.
    """
    radius = 6
    width = 1

    # Generate random angles
    np.random.seed(seed)
    angles = np.random.uniform(0, 2 * np.pi, num_points)

    # Calculate x and y coordinates for the ring shape
    x_ring = (radius + np.random.normal(scale=width, size=num_points)) *\
        np.cos(angles)
    y_ring = (radius + np.random.normal(scale=width, size=num_points)) *\
        np.sin(angles)

    # Generate x and y coordinates for the standard-shaped cloud
    x_standard = np.random.normal(size=num_points)
    y_standard = np.random.normal(size=num_points)

    # Join all the points. Blues are codified as 1 and reds as 0
    blue_dots = np.array([x_standard, y_standard, [1] * len(x_standard)])
    red_dots = np.array([x_ring, y_ring, [0] * len(x_standard)])
    all_dots = np.concatenate((red_dots, blue_dots), axis=1).T

    # Shuffle the points
    np.random.shuffle(all_dots)

    # Generate a grid of points for the background color
    # Get the max and min of x and y
    x_min = np.min(x_ring)
    x_max = np.max(x_ring)
    y_min = np.min(y_ring)
    y_max = np.max(y_ring)

    # Generate x and y coordinates for the grid of points
    x_coords = np.arange(x_min, x_max, 0.2)
    y_coords = np.arange(y_min, y_max, 0.2)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    length = x_grid.shape[0] * x_grid.shape[1]

    # Flatten the grid of points into a 1D array of x,y coordinates
    tst = np.vstack([x_grid.ravel(), y_grid.ravel(), [0] * length])

    return tst.astype(dtype=np.float32), all_dots.T.astype(dtype=np.float32)

# %%

def my_plot(slim_points, fat_points=np.array([])):
    """Dedicated function to display points"""

    """
    This function receives two npa of dimensions (3, n), where n is the number
    of dots to represent. The second npa is optional.
    In row 0 there is the x coordinate, in row 1 there is the y coordinate, and
    in row 2 there is the label (0 or 1). If the label is 0 the dot will be red,
    and if it is 1 it will be blue.

    The dots of the first block are small points, while those of the second
    are big dots. It is intended so that the first np-array is the set of points
    that make up a grid under the image. Their colors are those indicated
    according to the output of the network we want to evaluate.
    The points of the second block are the original ones, with their original
    colors.
    """

    xmin = np.min(slim_points, axis=1)[0]
    xmax = np.max(slim_points, axis=1)[0]
    ymin = np.min(slim_points, axis=1)[1]
    ymax = np.max(slim_points, axis=1)[1]

    sp_blue = slim_points[:2, np.argwhere(slim_points[2, :] == 1)].squeeze()
    sp_red = slim_points[:2, np.argwhere(slim_points[2, :] == 0)].squeeze()

    if fat_points.size != 0:
        xmin = min(xmin, np.min(fat_points, axis=1)[0])
        xmax = max(xmax, np.max(fat_points, axis=1)[0])
        ymin = min(ymin, np.min(fat_points, axis=1)[1])
        ymax = max(ymax, np.max(fat_points, axis=1)[1])

        fp_blue = fat_points[:2, np.argwhere(fat_points[2, :] == 1)].squeeze()
        fp_red = fat_points[:2, np.argwhere(fat_points[2, :] == 0)].squeeze()

    rgx = xmax - xmin
    rgy = ymax - ymin

    # Plot the graphic with dots
    plt.plot(sp_blue[0], sp_blue[1], color='#5555FF', linewidth=0, marker='o',
             markersize=4, alpha=1)
    plt.plot(sp_red[0], sp_red[1], color='#FF5555', linewidth=0, marker='o',
             markersize=4, alpha=1)
    if fat_points.size != 0:
        plt.plot(fp_blue[0], fp_blue[1], color="blue", linewidth=0,
                 marker='o', markersize=10)
        plt.plot(fp_red[0], fp_red[1], color="red", linewidth=0,
                 marker='o', markersize=10)
    plt.ylim(ymin - 0.1 * rgy, ymax + 0.1 * rgy)
    plt.xlim(xmin - 0.1 * rgx, xmax + 0.1 * rgx)
    plt.show()



# %%

def batch_it(n, dots):
    """
    Helper function to split the data in batches. Specific for this lab.
    """
    inputs = torch.from_numpy(dots[:2, :]).reshape(2, n, -1).permute(2, 1, 0)
    labels = torch.from_numpy(dots[2, :]).reshape(1, n, -1).permute(2, 1, 0)

    return inputs, labels


# %%

def acc(prediction, ground_truth):
    # Accuracy is calculated
    currect_preds = torch.sum(torch.eq(prediction.cpu(),
                                        ground_truth.reshape(-1, 1)))
    acc = currect_preds.item() / torch.numel(ground_truth)
    print("Acc: {:.4f}".format(100 * acc))    


