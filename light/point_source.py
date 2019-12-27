import numpy as np
import light.light_source as light_source
import light.photon as photon

class PointSource(light_source.Light):


    def __init__(self,x0,y0,z0,T=220,ax=None):


        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.T = T

        super().__init__(x0, y0, z0,T,ax)

        ax.scatter(x0, y0, z0, marker="o", color="black")


        return None

    def generate_photons(self,N_photons):
        photons=[]

        # Generate N photons around the light source
        for k in range(N_photons):
            x0, y0, z0 = self.x0, self.y0,self.z0
            theta, phi = np.random.rand()*np.pi,np.random.rand()*2.0*np.pi
            wavelength = 500 # There should be a function to take care of this
            photons.append(photon.Photon(x0=x0,y0=y0,z0=z0,theta=theta,phi=phi,wavelength=wavelength))

        return photons
