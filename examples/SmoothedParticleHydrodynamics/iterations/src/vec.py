# Written by Matthew Leach 

import math

class vec(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def dot(v1, v2):
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

    @staticmethod
    def cross(v1, v2):
        x = v1.y * v2.z - v1.z * v2.y
        y = v1.z * v2.x - v1.x * v2.z
        z = v1.x * v2.y - v1.y * v2.x
        return vec(x, y, z)

    @staticmethod
    def normalize(v):
        return v / v.norm()

    def norm(self):
        return math.sqrt(vec.dot(self, self))

    def __add__(self, v):
        return vec(self.x + v.x, self.y + v.y, self.z + v.z)
		
    def __sub__(self, v):
		return self + (-v)

    def __neg__(self):
        return vec(-self.x, -self.y, -self.z)

    def __mul__(self, v):
        if isinstance(v, vec):
            return vec(self.x * v.x, self.y * v.y, self.z * v.z)
        else:
            return vec(self.x * v, self.y * v, self.z * v)

    def __rmul__(self, v):
        return self.__mul__(v)

    def __div__(self, v):
        if isinstance(v, vec):
            return vec(self.x / v.x, self.y / v.y, self.z / v.z)
        else:
            return vec(self.x / v, self.y / v, self.z / v)
