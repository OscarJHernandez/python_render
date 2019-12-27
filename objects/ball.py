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
        ax.plot_wireframe(x, y, z, color="r", alpha=1.0)


        return None

    def check_collision(self,p1):
        collision = False

        # Compute the distance
        d = utils.distance(p1,self.p0)

        if(d<=self.radius):
            collision = True

        return collision

    def return_name(self):
        return self.name


#ball = Ball()