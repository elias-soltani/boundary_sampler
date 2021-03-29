"""
This script reads a point clouds in csv format and extracts and outputs the boundary points.
"""

import csv
import math
import numpy as np
from scipy import spatial
import time


def outline(points):
    ranges = [min(p[0] for p in points), max(p[0] for p in points),
              min(p[1] for p in points), max(p[1] for p in points),
              min(p[2] for p in points), max(p[2] for p in points)]
    extends = [abs(ranges[i+1]-ranges[i]) for i in range(0, 6, 2)]
    return ranges, extends


def centre_radius(ranges, extends):
    centre = [(ranges[i] + ranges[i+1])/2 for i in range(0, 6, 2)]
    major_diameter = max(extends[0], extends[1])
    minor_diameter = min(extends[0], extends[1])
    radius = major_diameter/2 * 1.1
    return centre, radius


def generate_sample_points(points, sample_along, sample_around):
    ranges, extends = outline(points)
    centre, radius = centre_radius(ranges, extends)
    along_z = sample_along
    along_theta = sample_around
    element_height = extends[2]/along_z
    x = []
    y = []
    for idx in range(along_theta):
        theta = idx * 2*math.pi/along_theta
        x.append(round(centre[0] + radius*math.cos(theta), 3))
        y.append(round(centre[1] + radius*math.sin(theta), 3))

    sample_points = []
    for n3 in range(along_z):
        circle_points = []
        for n2 in range(along_theta):
            circle_points.append([x[n2], y[n2], ranges[4]+element_height*n3])
        sample_points.append(circle_points)
    sample_points = np.array(sample_points)

    shape = sample_points.shape
    sample_points2 = sample_points.reshape(shape[0] * shape[1], 3)
    sampled = points[spatial.cKDTree(points).query(sample_points2)[1]]

    return sampled


def main(input_file, **kwargs):
    output_file = kwargs["output_file"]
    sample_along = kwargs["sample_along"] if kwargs["sample_along"] else 30
    sample_around = kwargs["sample_around"] if kwargs["sample_around"] else 30
    with open(input_file, 'r') as f1:
        reader = csv.reader(f1)
        next(reader, None)
        points = []
        for point in reader:
            points.append([float(d) for d in point])

    points = np.array(points)
    sampled = generate_sample_points(points, sample_along, sample_around)

    with open(output_file, 'w', newline='') as f2:
        writer = csv.writer(f2)
        writer.writerow(['x', 'y', 'z'])
        writer.writerows(sampled)


if __name__ == "__main__":
    input_file = r"input/heart.csv"
    output_file = r"expected_results/sampled.csv"
    sample_along = 5
    sample_around = 32
    start = time.time()
    main(input_file, output_file=output_file, sample_along=sample_along, sample_around=sample_around)
    end = time.time()
    print(end - start)
