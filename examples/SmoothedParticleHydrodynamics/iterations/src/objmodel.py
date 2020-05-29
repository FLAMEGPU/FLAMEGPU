from vec import *

# Read all triangles of a .obj model
def loadObjectModel(filePath):
	verts = []
	tris = []
	vcount = 0;
	tricount = 0	

	with open(filePath, 'r') as f:
		for line in f:
			if line[0] == 'v':
				words = line.rstrip().split(' ')
				verts.append(vec(float(words[1]), float(words[2]), float(words[3])))
				vcount = vcount + 1
			if line[0] == 'f':
				words = line.rstrip().split(' ')
				# -1 as verts are 1-indexed in obj format
				i1 = words[1]
				i2 = words[2]
				i3 = words[3]
				i1 = int(i1)
				i2 = int(i2)
				i3 = int(i3)
				v1 = verts[i1-1]
				v2 = verts[i2-1]
				v3 = verts[i3-1]
				tris.append([v1, v2, v3])
				tricount = tricount + 1
	return tris