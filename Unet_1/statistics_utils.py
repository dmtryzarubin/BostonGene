import numpy as np
from scipy.spatial import distance


def calc_mean_location(objects):
    """
    Calculate mean location of each object on the image grid
    """
    mean_loc = {}
    for object in objects:
        mean_loc[object] = np.mean(objects[object], axis=0)
    
    return mean_loc


def calc_distance(objects):
    """
    Calculates eucledian distance between each object
    """
    items = np.array(list(objects.keys()))
    distances = {}
    for object1 in items:
        for object2 in items[np.where(items != object1)]:
            d = 0
            between = f'{object1} | {object2}'
            distances[between] = []
            for idx, _ in enumerate(objects[object1]):
                d = distance.euclidean(objects[object1][idx], objects[object2][idx])
                between = f'{object1} | {object2}'
                distances[between].append(int(d))
    return distances


def calc_mean_distance(distances):
    """
    Calculates mean distance between each instance
    """
    mean_dist = {}
    for pair in distances:
        mean_dist[pair] = np.mean(distances[pair])
    
    return mean_dist


def calc_location(data_set):
    """
    Calculates center coordinates of each object in data_set
    """

    objects = {
        'filled_square' : [],
        'filled_circle' : [],
        'traingle' : [],
        'circle' : [],
        'mesh_square' : [],
        'plus' : []
    }

    for idx, _ in enumerate(data_set):
        masks = data_set[idx][1]
        num_objs = len(masks)
        for i in range(num_objs):
            object = list(objects.keys())[i]
            if np.max(masks[i]) == 0:
                objects[object].append([masks.shape[2] // 2, masks.shape[2] // 2])
            else:
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                objects[object].append([np.mean([xmin, xmax]), np.mean([ymin, ymax])])
    
    return objects