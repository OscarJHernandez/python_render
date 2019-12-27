import numpy as np



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
