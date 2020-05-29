from vec import *


def pointInTriangle(p, v1, v2 ,v3):
    u = v2 - v1
    v = v3 - v1
    n = vec.cross(u, v)
   
    w = p - v1

    gamma = vec.dot(vec.cross(u,w),n) / vec.dot(n, n);    
    beta = vec.dot(vec.cross(w,v), n) / vec.dot(n, n);
    alpha = 1 - gamma - beta;
    
    return ((0 <= alpha) and (alpha <= 1) and
            (0 <= beta)  and (beta  <= 1) and
            (0 <= gamma) and (gamma <= 1))

def rayTriangleIntersection(r0, r, v1, v2, v3):
	intersection = vec(0.0, 0.0, 0.0)
	# Test if ray intersects plane triangle lies in
	# Plane defined by (p-p0).n = 0
	# p0 can be any point in the triangle, e.g. any of the vertices, we choose as v1
	# Also need n, given by cross product of two edges of triangle, i.e. v1->v2, v1->v3 then normalized

	p0 = v1
	u = v2 - v1
	v = v3 - v1
	n = vec.cross(v2-v1, v3-v1)
	n = vec.normalize(n)

	# Equation of ray is p = r0 + rd
	# Sub into plane equation
	# => ((r0 + rd) - p0).n = 0
	# => (r0 - p0).n + d(r.n) = 0
	# => d = (p0-r0).n / r.n

	r0p0 = p0 - r0
	if (vec.dot(r, n) == 0):
		return (False, intersection)
	d = vec.dot(r0p0, n) / vec.dot(r, n)
	

	# If d = 0 ray and plane are parallel -> ignore
	# Otherwise, we have an intersection at r0 + dr
	if (d != 0.0):
		intersection = r0 + d * r
		if pointInTriangle(intersection, v1, v2, v3):
			return (True, intersection)
		else:
			return (False, intersection)