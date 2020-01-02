import numpy as np
import light.photon as photon


class Object():
    '''
    The class that defines objects

    '''

    def __init__(self,x0=0,y0=0,z0=0):
        '''
        The constructor for the class
        '''

        # Store the position of the object
        self.position = np.asarray([x0,y0,z0])

        return None

    def get_position(self):

        return self.position

    def check_collision(self,p0):
        '''
        Checks collision of a point p0

        :return:
        '''

        collision = False


        return collision

    def resolve_collision(self,incident_photon):

        # Get the current position of the photon
        xp,yp,zp = incident_photon.return_position()
        p1 = np.asarray([xp,yp,zp])

        # Determine the normal of the surface at the collision point
        normal = self.return_normal_vector(p1)

        # Get the cartesian unit vector of the photon vector
        photon_velocity = incident_photon.return_velocity_unit_vector()
        #vx0,vy0,vz0= photon_velocity

        reflected_velocity = -photon_velocity +2.0*(photon_velocity.dot(normal))*normal
        vx0, vy0, vz0 = reflected_velocity
        reflected_wavelength = self.reflected_wavelength(initial_wavelength=incident_photon.wavelength)

        # Reflected photon
        photon_reflected = photon.Photon(x0=xp, y0=yp, z0=zp, vx0=vx0,vy0=vy0,vz0=vz0, wavelength=reflected_wavelength, path=incident_photon.path, time=incident_photon.time,reflected=True)

        # Transmitted photon


        # resolve the collision change the properties of the photon


        return photon_reflected
