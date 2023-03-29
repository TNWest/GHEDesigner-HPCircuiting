from math import atan, pi, sin, sqrt

import numpy as np


class Shapes:
    """
    a class to represent nogo zones

    Attributes
    ----------
    cx : float
        the x value of the centroid
    cy : float
        the y value of the centroid
    xw: float
        the xwidth
    yw : float
        the ywidth
    theta: float
        the rotation of the shape
    c00: [float,float]
        x,y location of 1 vertex
    c01: [float,float]
        x,y location of 2nd vertex
    c10: [float,float]
        x,y location of 3rd vertex
    c11: [float,float]
        x,y location of 4th vertex
    c: [[float]]
        array containing x,y locations of vertices
    maxy : float
        maximum y value of shape
    miny: float
        minimum y value of shape
    maxx: float
        maximum x value of shape
    minx: float
        minimum x value of shape
    Methods
    -------
    lineintersect(xy)
        determines the intersection of this shape and the given line segment
    pointintersect(xy)
        determines whether the given point is inside of the rectangle
    """

    def __init__(self, c):
        """
         contructs a shape object

        Parameters
        ----------
        :param cx: float
            the x location of the centroid
        :param cy: float
            the y location of the centroid
        :param xw: float
            the width in the x dir
        :param yw: float
            the width in the y dir
        :param theta: float
            the amount of rotation in radians
        :param sh: string
            string specifying the desired shape supports:
            B,S,L,U,T,BL
        """
        self.c = np.array(c)
        # print(c)
        xs = [0] * len(self.c)
        ys = [0] * len(self.c)
        for i in range(len(self.c)):
            xs[i] = self.c[i][0]
            ys[i] = self.c[i][1]
        self.maxx = max(xs)
        self.minx = min(xs)
        self.maxy = max(ys)
        self.miny = min(ys)





    def lineintersect(self, xy, rotate=0, intersection_tolerance=1e-12):
        """
        returns the intersections between a line segment and the shape

        Parameters
        -----------
        :param xy: [float,float,float,float]
            the x,y values of both endpoints of the line segment
        :return: [[float]]
            the x,y values of the intersections
        """
        x1, y1, x2, y2 = xy

        # Check that the line is in the neighboorhood of the polygon.
        max_x = max(x1, x2)
        min_x = min(x1, x2)
        max_y = max(y1, y2)
        min_y = min(y1, y2)
        if min_x > self.maxx or max_x < self.minx or \
           min_y > self.maxy or max_y < self.miny:
           return []

        rA = []
        for i in range(len(self.c)):
            if i == len(self.c) - 1:
                c1 = self.c[len(self.c) - 1]
                c2 = self.c[0]
                r = vectorintersect(
                    [c1[0], c1[1], c2[0], c2[1]],
                    [x1, y1, x2, y2],
                    intersection_tolerance,
                )
                # print(r)
                if len(r) == 1:
                    r = r[0]
                    if not point_within_bounding_box(xy,
                           [c1[0], c1[1], c2[0], c2[1]], r, intersection_tolerance):
                        continue
                    rA.append(r)
                elif len(r) == 2:
                    rA.append(r[0])
                    rA.append(r[1])
            else:
                c1 = self.c[i]
                c2 = self.c[i + 1]
                r = vectorintersect(
                    [c1[0], c1[1], c2[0], c2[1]],
                    [x1, y1, x2, y2],
                    intersection_tolerance,
                )
                if len(r) == 1:
                    r = r[0]
                    if not point_within_bounding_box(xy,
                           [c1[0], c1[1], c2[0], c2[1]], r, intersection_tolerance):
                        continue
                    rA.append(r)
                elif len(r) == 2:
                    rA.append(r[0])
                    rA.append(r[1])
        # print("x value: %f, r values:"%x1)
        # print(rA)

        rA = sortIntersections(rA, rotate)

        existing_point_array = [[x1, y1], [x2, y2]]
        existing_point_array.extend(self.c)
        points_to_return = []
        for point in rA:
            point_already_exists = False
            for existing_point in existing_point_array:
                dist = sqrt((point[0] - existing_point[0])**2 + (point[1] - existing_point[1])**2)
                if dist < intersection_tolerance:
                    point_already_exists = True
            if not point_already_exists:
                points_to_return.append(point)
        # print(rA)
        return points_to_return

    def pointintersect(self, xy):
        """
        returns whether the given point is inside of the shape

        Parameters
        -----------
        :param xy: [float,float]
            x,y value of point
        :return: boolean
            true if inside, false if not
        """
        x, y = xy
        if (x > self.maxx or x < self.minx) or (y > self.maxy or y < self.miny):
            # print("Returning False b/c outside of box")
            return False
        left_x = self.minx - 10
        inters = self.lineintersect([left_x, y, x, y])
        # print(inters)
        inters = [inter for inter in inters if inter[0] <= x]
        # print("x: %f"%x,inters)
        if len(inters) == 1:
            # print("Returning True")
            return True
        i = 0
        while i < len(inters):
            for vert in self.c:
                if inters[i][0] == vert[0] and inters[i][1] == vert[1]:
                    inters.pop(i)
                    i -= 1
                    break
            i += 1
        if len(inters) % 2 == 0:
            # print(len(inters))
            return False
        else:
            # print("returning True")
            return True

    def getArea(self):
        """
        returns area of shape
        :return: float
            area of shape
        """
        sum = 0
        for i in range(len(self.c)):
            if i == len(self.c) - 1:
                sum += self.c[len(self.c) - 1][0] * self.c[0][1] - (
                    self.c[len(self.c) - 1][1] * self.c[0][0]
                )
                continue
            sum += self.c[i][0] * self.c[i + 1][1] - (self.c[i][1] * self.c[i + 1][0])
        return 0.5 * sum

def point_within_bounding_box(line_segment_1, line_segment_2, point, intersection_tolerance):

    xp, yp = point
    x11, y11, x12, y12 = line_segment_1
    x21, y21, x22, y22 = line_segment_2

    # Check that point is in neighborhood of line segment 1.
    max_x_1 = max(x11, x12)
    min_x_1 = min(x11, x12)
    max_y_1 = max(y11, y12)
    min_y_1 = min(y11, y12)
    if  xp - max_x_1 > intersection_tolerance \
        or xp - min_x_1 < -intersection_tolerance\
        or yp - max_y_1 > intersection_tolerance\
        or yp - min_y_1 < -intersection_tolerance:
        return False

    # Check that point is in neighborhood of line segment 2.
    max_x_2 = max(x21, x22)
    min_x_2 = min(x21, x22)
    max_y_2 = max(y21, y22)
    min_y_2 = min(y21, y22)
    if xp - max_x_2 > intersection_tolerance \
            or xp - min_x_2 < -intersection_tolerance \
            or yp - max_y_2 > intersection_tolerance \
            or yp - min_y_2 < -intersection_tolerance:
        return False

    return True


def sortIntersections(rA, rotate):
    if len(rA) == 0:
        return rA
    vals = [0] * len(rA)
    i = 0
    for inter in rA:
        phi = 0
        if inter[0] == 0:
            phi = pi / 2
        else:
            phi = atan(inter[1] / inter[0])
        R = sqrt(inter[1] * inter[1] + inter[0] * inter[0])
        refang = pi / 2 - phi
        # sign = 1
        if phi > pi / 2:
            if phi > pi:
                if phi > 3 * pi / 2.0:
                    refang = 2 * pi - phi
                else:
                    refang = 3.0 * pi / 2.0 - phi
            else:
                refang = pi - phi
        # if phi > pi/2 + rotate and phi < 3*pi/2 + rotate:
        # sign = -1
        vals[i] = R * sin(refang + rotate)
        i += 1
    zipped = zip(vals, rA)
    zipped = sorted(zipped)
    rA = [row for _, row in zipped]
    return rA


def vectorintersect(l1, l2, intersection_tolerance):
    """
     gives the intersection between two line segments

    Parameters
    -----------
    :param l1: [[float]]
        endpoints of first line segment
    :param l2: [[float]]
        endpoints of the second line segment
    :return: [float,float]
        x,y values of intersection (returns None if there is none)
    """
    x11, y11, x12, y12 = l1
    x21, y21, x22, y22 = l2

    # Check if segments are in the same neighborhood.
    if min(x11, x12) > max(x21, x22) or min(x21, x22) > max(x11, x12) \
       or min(y11, y12) > max(y21, y22) or min(y21, y22) > max(y11, y12):
        return []

    if abs(x12 - x11) <= intersection_tolerance:
        a1 = float("inf")
    else:
        a1 = (y12 - y11) / (x12 - x11)
        c1 = y11 - x11 * a1
    if abs(x22 - x21) <= intersection_tolerance:
        a2 = float("inf")
    else:
        a2 = (y22 - y21) / (x22 - x21)
        c2 = y21 - x21 * a2
    if a1 == float("inf") or a2 == float("inf"):
        if a1 == float("inf") and a2 == float("inf"):
            if abs(x11 - x21) < intersection_tolerance:
                return [[x11, y11], [x12, y12]]
            else:
                return []
        elif a1 == float("inf"):
            rx = x11
            ry = a2 * x11 + c2
            return [[rx, ry]]
        else:
            rx = x21
            ry = a1 * x21 + c1
            return [[rx, ry]]
    if abs(a1 - a2) <= intersection_tolerance:
        if abs(y22 - (a1 * x22 + c1)) <= intersection_tolerance:
            return []
        else:
            return []
    rx = (c2 - c1) / (a1 - a2)
    ry = a1 * (c2 - c1) / (a1 - a2) + c1

    return [[rx, ry]]

