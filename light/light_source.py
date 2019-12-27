import light.photon as photon
import numpy as np

class Light:

	def __init__(self,x0,y0,z0,T=220,ax=None):
		'''
		Instantiates the light source at a
		specific location with a given temperature
		'''
		self.p0 = np.asarray([x0,y0,z0])
		self.T = T

		return None

	def generate_photons(self):
		'''
		Generates photons that will be propagated

		:param self:
		:return:
		'''
		photons = []

		return photons