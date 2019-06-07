import os
import sys
import termcolor
import numpy as np
from plyfile import PlyData, PlyElement


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    return np.stack([x,y,z], axis=1)


def read_label_ply(filename):
    plydata = PlyData.read(filename)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    label = np.asarray(plydata.elements[0].data['label'])
    return np.stack([x,y,z], axis=1), label


def read_color_ply(filename):
    plydata = PlyData.read(filename)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    r = np.asarray(plydata.elements[0].data['red'])
    g = np.asarray(plydata.elements[0].data['green'])
    b = np.asarray(plydata.elements[0].data['blue'])
    return np.stack([x,y,z,r,g,b], axis=1)


def read_txt(filename):
    # Return a list
    res= []
    with open(filename) as f:
        for line in f:
            res.append(line.strip())
    return res


def read_label_txt(filename):
    # Return a list
    res= []
    with open(filename) as f:
        for line in f:
            res.append(int(line.strip()))
    res = np.array(res)
    return res


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


def write_color_ply(points, filename, text=True):
    points = [(points[i,0], points[i,1], points[i,2], points[i,3], points[i,4], points[i,5]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


def write_label_txt(label, filename):
    f = open(filename, 'w')
    for i in range(np.shape(label)[0]):
        f.write('{0}\n'.format(label[i]))


# convert to colored strings
def toRed(content): return termcolor.colored(content, "red", attrs=["bold"])
def toGreen(content): return termcolor.colored(content, "green", attrs=["bold"])
def toBlue(content): return termcolor.colored(content, "blue", attrs=["bold"])
def toCyan(content): return termcolor.colored(content, "cyan", attrs=["bold"])
def toYellow(content): return termcolor.colored(content, "yellow", attrs=["bold"])
def toMagenta(content): return termcolor.colored(content, "magenta", attrs=["bold"])



