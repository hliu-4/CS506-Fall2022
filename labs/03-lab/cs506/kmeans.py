from collections import defaultdict
from math import inf
import random
import csv
from numpy.random import choice

def get_centroid(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)
    
    Returns a new point which is the center of all the points.
    """
    if len(points) == 0:
        return []
    
    pt = [0 for i in range(len(points[0]))]

    for i in range(len(points)):
        for j in range(len(points[0])):
            pt[j] += points[i][j]
        
    return [x/len(points) for x in pt]


def get_centroids(dataset, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the centroid for each of the assigned groups.
    Return `k` centroids in a list
    """
    if len(dataset) == 0 or len(assignments) == 0:
        return []
    
    k = len(set(assignments))
    clusters = [[] for i in range(k)]
    centroids = []

    for i in range(len(dataset)):
        clusters[assignments[i]].append(dataset[i])
    
    for i in range(k):
        centroids.append(get_centroid(clusters[i]))
    
    return centroids


def assign_points(data_points, centers):
    """
    """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    res = 0

    for i in range(len(a)):
        res += (a[i] - b[i])**2
    
    return res**(1/2)


def distance_squared(a, b):
    return distance(a,b)**2


def cost_function(clustering):
    centroids = []
    
    for i in range(len(clustering)):
        centroids.append(get_centroid(clustering[i]))

    cost = 0
    
    for i in range(len(clustering)):
        for pt in clustering[i]:
            cost += distance_squared(centroids[i], pt)
    
    return cost




def generate_k(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    pts = []

    sett = [i for i in range(len(dataset))]

    for i in range(k):
        num = random.randint(0, len(sett)-1)
        pts.append(dataset[sett[num]])
        prev = sett[num]
        sett.remove(prev)

    return pts


def generate_k_pp(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    where points are picked with a probability proportional
    to their distance as per kmeans pp
    """
    if k == 0:
        return []
    
    sett = [i for i in range(len(dataset))]

    pts = []

    num = random.randint(0, len(sett)-1)
    pts.append(dataset[sett[num]])
    prev = sett[num]
    sett.remove(num)

    mindist = {}
    for i in range(1,k):
        # assigning proportions
        total = 0
        for data in range(len(sett)):
            dist = distance(dataset[sett[data]], dataset[prev])
            if i == 1:
                mindist[sett[data]] = dist
                total += dist
            else:
                mindist[sett[data]] = min(dist, mindist[sett[data]])
                total += min(dist, mindist[sett[data]])
        prop = [mindist[d]/total for d in mindist]
        
        # get random point
        prev = choice(sett, 1, p=prop)[0]
        pts.append(dataset[prev])
        sett.remove(prev)
        del mindist[prev]

    return pts

        


def _do_lloyds_algo(dataset, k_points):
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = get_centroids(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering


def k_means(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")
    
    k_points = generate_k(dataset, k)
    return _do_lloyds_algo(dataset, k_points)


def k_means_pp(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k_pp(dataset, k)
    return _do_lloyds_algo(dataset, k_points)
