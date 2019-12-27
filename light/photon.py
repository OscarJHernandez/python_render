import numpy as np
import utils as utils

class Photon:
    '''
    The photon class
    '''

    def __init__(self, x0, y0, z0, theta, phi,wavelength=600):


        # The wavelength of the photon
        self.wavelength = wavelength # In nano meters

        # The position
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

        # The direction of the ray
        self.c = 1.0
        self.theta = theta
        self.phi = phi

        # The trajectory
        self.path = []
        self.path.append([x0,y0,z0])

        # The parametrized time
        self.time = []
        self.time.append([0.0])

        return None

    def step_forward(self, dt):
        c = self.c

        # velocity vector
        vx0 = np.sin(self.theta) * np.cos(self.phi)
        vy0 = np.sin(self.theta) * np.sin(self.phi)
        vz0 = np.cos(self.theta)

        self.x0 = self.x0 + c * vx0 * dt
        self.y0 = self.y0 + c * vy0 * dt
        self.z0 = self.z0 + c * vz0 * dt

        # Add the path to the trajectory
        self.path.append([self.x0,self.y0,self.z0])

        # Retrive the current time
        current_time = self.time[-1][0]

        self.time.append([current_time+dt])


        return None

    def return_position(self):
        return self.x0, self.y0, self.z0

    def return_path(self):
        '''

        :return:
        '''

        x_path = np.asarray(self.path)[:,0]
        y_path = np.asarray(self.path)[:,1]
        z_path = np.asarray(self.path)[:,2]
        t_path = np.asarray(self.time)[:]

        return x_path,y_path,z_path,t_path

    def return_color(self):

        color = utils.wavelength_to_rgb(self.wavelength,gamma=1.0)


        return color

    def plot_photon(self,ax):
        '''
        plots the photon

        :param ax:
        :return:
        '''

        x_path, y_path, z_path, t_path = self.return_path()
        photon_color = self.return_color()
        ax.plot(x_path,y_path,z_path,color=photon_color,alpha=0.1)
        return None

