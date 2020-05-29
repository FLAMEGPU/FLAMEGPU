import numpy as np
from raycast import *
from objmodel import *
from vec import *

class ParticleGenerator(object):
	def __init__(self):
		self.rayz = 0.5
		self.rayDirection = vec(0, 0, -1)
		self.particles = []
		self.staticParticles = []
		self.rayOrigins = []
		
	def generateParticles(self, model, isStatic):
		print "Beginning particle generation\n"
		self.generateRayOrigins(isStatic)
		
		totalRays = len(self.rayOrigins)
		rayCounter = 0
		lastPercentage = -1
		
		for rayOrigin in self.rayOrigins:			
			rayCounter = rayCounter + 1
			percentageComplete = int(100*float(rayCounter)/float(totalRays))
			if percentageComplete != lastPercentage:
				lastPercentage = percentageComplete
				print ("Ray casting " + str(percentageComplete) + "% complete")
			intersections = []
			for tri in model:	
				result, intersection = rayTriangleIntersection(rayOrigin, self.rayDirection, tri[0], tri[1], tri[2])
				if result:
					intersections.append(intersection)
		
			if (intersections):
				# Fill volume with fluid particles
				
				# Sort intersections by z-values and generate pairs of intersections
				intersections.sort(key=lambda x:x.z)
				intersections = zip(*(iter(intersections),)*2)

				for pair in intersections:
					# Generate particles between the two intersections of each pair
					zstart = pair[0].z
					zend = pair[1].z
					numzvals = int((zend - zstart) / 0.0125)
					zvals = np.linspace(zstart, zend, numzvals)
					for zval in zvals:
						if isStatic:
							self.staticParticles.append(vec(rayOrigin.x, rayOrigin.y, zval))
						else:
							self.particles.append(vec(rayOrigin.x, rayOrigin.y, zval))
							
	def generateRayOrigins(self, isStatic):
		print "Generating ray origins\n"
		self.rayOrigins = []
		if isStatic:
			rayDensity = 160
		else:
			rayDensity = 80
			
		for x in np.linspace(-0.5, 0.5, rayDensity):
			for y in np.linspace(-0.5, 0.5, rayDensity):
				self.rayOrigins.append(vec(x, y, self.rayz))	
				
	def save(self):
		print "Saving..."
		with open("0.xml", 'w') as f:
			f.write('<states><itno>0</itno><environment></environment>')
			
			dx = dy = dz = 0.0
			id = 0
			print "Writing " + str(len(self.particles)) + " dynamic fluid particles"
			for particle in self.particles:

				f.write('<xagent><name>FluidParticle</name><id>' + str(id) + '</id>')
				f.write('<x>' + str(particle.x) + '</x>')
				f.write('<y>' + str(particle.y) + '</y>')
				f.write('<z>' + str(particle.z) + '</z>')
				f.write('<dx>' + str(dx) + '</dx>')
				f.write('<dy>' + str(dy) + '</dy>')
				f.write('<dz>' + str(dz) + '</dz>')
				f.write('<fx>' + str(dx) + '</fx>')
				f.write('<fy>' + str(dy) + '</fy>')
				f.write('<fz>' + str(dz) + '</fz>')
				f.write('<isStatic>0</isStatic>')
				f.write('</xagent>')
				id = id + 1
				
				
			print "Writing " + str(len(self.staticParticles)) + " static particles"
			for particle in self.staticParticles:

				f.write('<xagent><name>FluidParticle</name><id>' + str(id) + '</id>')
				f.write('<x>' + str(particle.x) + '</x>')
				f.write('<y>' + str(particle.y) + '</y>')
				f.write('<z>' + str(particle.z) + '</z>')
				f.write('<dx>' + str(dx) + '</dx>')
				f.write('<dy>' + str(dy) + '</dy>')
				f.write('<dz>' + str(dz) + '</dz>')
				f.write('<fx>' + str(dx) + '</fx>')
				f.write('<fy>' + str(dy) + '</fy>')
				f.write('<fz>' + str(dz) + '</fz>')
				f.write('<isStatic>1</isStatic>')
				f.write('</xagent>')
				id = id + 1
				
			f.write('</states>')
		print "Particles saved"