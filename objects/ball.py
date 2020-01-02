import numpy as np
import objects.object as object
import utils as utils


class Ball(object.Object):

    def __init__(self,x0=0,y0=0,z0=0,radius=1,ax=None):
        '''
        Instantiate the ball class
        :param x0: x-coord
        :param y0: y-coord
        :param z0: z-coord
        '''

        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.p0 = np.asarray([x0,y0,z0])
        self.name = 'Ball'

        super().__init__(x0,y0,z0)
        self.radius = radius

        # Draw the ball
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:20j]
        x = self.x0 + self.radius*np.cos(u) * np.sin(v)
        y = self.y0 + self.radius*np.sin(u) * np.sin(v)
        z = self.z0 + self.radius*np.cos(v)

        if(ax!= None):
            ax.plot_wireframe(x, y, z, color="r", alpha=0.0)


        return None

    def return_normal_vector(self,p1):
        '''
        :return:
        '''

        # Get the coordinates
        x1,y1,z1 = p1

        # Compute the normal vector at the point p1 for the sphere
        nx = (self.x0-x1)/np.sqrt(self.radius**2-(self.x0-x1)**2-(self.y0-y1)**2)

        ny = (self.y0-y1)/np.sqrt(self.radius**2-(self.x0-x1)**2-(self.y0-y1)**2)

        nz = -1

        normal = np.asarray([nx,ny,nz])/np.sqrt(nx**2+ny**2+nz**2)

        return normal

    def reflected_wavelength(self,initial_wavelength):


        wavelength = 650

        return wavelength

    def check_collision(self,p1):
        '''

        :param p1: the target point
        :return:
        '''
        tol = 1e-2

        collision = False

        # Compute the distance
        # p1: location of the target point
        d = utils.distance(p1,self.p0)

        if( abs(d-self.radius) < tol ):
            collision = True

        return collision

    def return_name(self):
        return self.name


#ball = Ball()