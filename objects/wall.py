import numpy as np
import objects.object as object
import utils as utils


class Wall(object.Object):


    def __init__(self,x0,y0,z0,Lx,Ly,Lz,ax=None):


        super().__init__(x0, y0, z0)
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.p0 = np.asarray([x0,y0,z0])
        self.L = np.asarray([Lx,Ly,Lz])
        self.name='Wall'

        v1 = np.asarray([Lx,Ly,Lz])
        v2 = np.asarray([Lx,-Ly,Lz])

        n_surface = np.cross(v1, v2)
        n_surface = n_surface / np.linalg.norm(n_surface)

        self.normal = n_surface

        # ax+by+c*z +d = 0
        d = -self.p0.dot(n_surface)

        # create x,y
        x_min,x_max = self.x0-Lx, self.x0+Lx
        y_min,y_max = self.y0-Ly, self.y0+Ly
        z_min,z_max = self.z0-Lz, self.z0+Lz
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        # calculate corresponding z
        z = -(n_surface[0] * xx + n_surface[1] * yy + d) / n_surface[2]
        z[z > z_max] = np.nan
        z[z < z_min] = np.nan

        if(ax!=None):
            ax.plot_wireframe(xx, yy, z, color="purple", alpha=1.0)


        return None
    def check_collision(self,p1):
        collision = False
        tol = 1.0e-3

        x1,y1,z1 = p1

        if( z1< self.z0 and self.x0-self.Lx<x1< self.x0+self.Lx and self.y0-self.Ly<y1< self.y0+self.Ly):
            collision = True

        #if(x1> self.x0+self.Lx and x1 < self.x0-self.Lx and y1> self.y0+self.Ly and y1 < self.y0-self.Ly and z1 < self.z0):
         #   collision=True
         #   print(collision)


        return collision

    def reflected_wavelength(self, initial_wavelength):
        wavelength = 550

        return wavelength

    def return_normal_vector(self,p1=[0.0,0.0,0.0]):
        '''
        :param p1:
        :return: normal vector
        '''

        normal = self.normal

        return normal

    def return_name(self):
        return self.name