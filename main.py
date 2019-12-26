import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Ball:
	
	def __init__(self,x0,y0,z0,radius):
		self.x0 = x0
		self.y0 = y0
		self.z0 = z0
		self.radius = radius
		return None

class Photon:
	
	def __init__(self,x0,y0,z0,theta,phi):
		self.x0 = x0
		self.y0 = y0
		self.z0 = z0
		
		# The direction of the ray
		self.c = 1.0
		self.theta = theta
		self.phi = phi
		
		return None
	
	def step_forward(self,dt):
		c = self.c

		# velocity vector
		vx0 = np.sin(self.theta)*np.cos(self.phi)
		vy0 = np.sin(self.theta)*np.sin(self.phi)
		vz0 = np.cos(self.theta)
		
		self.x0 = self.x0+c*vx0*dt
		self.y0 = self.y0+c*vy0*dt
		self.z0 = self.z0+c*vz0*dt	
		return None
		
	def return_position(self):
		
		
		return self.x0,self.y0,self.z0



class Light:

	def __init__(self,x0,y0,z0,T=220):
		'''
		Instantiates the light source at a
		specific temperature
		'''
		self.r0 = np.asarray([x0,y0,z0])


		return None

class Camera:
	
	def __init__(self,x0,y0,z0,Ngrid):
		self.x0 = x0
		self.y0 = y0
		self.z0 = z0
		self.image = np.zeros((Ngrid,Ngrid))
		return None
		
	def convert_spherical_to_cartesian(self,theta,phi):

		print("theta: ", theta, "phi: ", phi)
		x = self.x0+np.sin(theta)*np.cos(phi)
		y = self.y0+np.sin(theta)*np.sin(phi)
		z = self.z0+np.cos(theta)

		p = np.zeros(3)
		p[0],p[1],p[2] = x,y,z

		print("x y z: ", x,y,z)
		print("")

		return p
		
	def direction(self,theta_min,theta_max,phi_min,phi_max):
		"""
		The direction and maximum angles the Camera can view
		
		this creates a screen that will represent what the camera 
		can see
		"""
		self.theta_min = theta_min
		self.theta_max = theta_max
		self.phi_min = phi_min
		self.phi_max = phi_max
		
		# Corners of the screen
		p_upper_left = self.convert_spherical_to_cartesian(theta_min,phi_min)
		p_upper_right = self.convert_spherical_to_cartesian(theta_min,phi_max)

		p_lower_left = self.convert_spherical_to_cartesian(theta_max,phi_min)
		p_lower_right = self.convert_spherical_to_cartesian(theta_max,phi_max)

		x1,y1,z1 = p_lower_left
		x2, y2, z2 = p_lower_right

		x3,y3,z3 = p_upper_left
		x4, y4, z4 = p_upper_right

		# The normal
		v1 = p_upper_left- p_lower_left
		v2 = p_lower_right - p_lower_left
		n_surface = np.cross(v1,v2)
		n_surface = n_surface/np.linalg.norm(n_surface)

		# ax+by+c*z +d = 0
		d  = -p_upper_right.dot(n_surface)

		# create x,y
		x_min = np.min([x1,x2,x3,x4])
		x_max = np.max([x1,x2,x3,x4])
		y_min = np.min([y1,y2,y3,y4])
		y_max = np.max([y1,y2,y3,y4])
		z_min = np.min([z1,z2,z3,z4])
		z_max = np.max([z1,z2,z3,z4])
		xx, yy = np.meshgrid(np.linspace(x_min,x_max,100) , np.linspace(y_min,y_max,100))

		# calculate corresponding z
		z = -(n_surface[0]*xx+n_surface[1]*yy +d)/n_surface[2]
		z[z>z_max] = np.nan
		z[z<z_min] = np.nan

		print("Z: ", z.shape)
		print("Z", z[0,0], xx[0,0],yy[0,0])

		ax.plot_surface(xx, yy, z, alpha=0.5)
		#======================================



		u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:20j]
		x = self.x0+ np.cos(u) * np.sin(v)
		y = self.y0+np.sin(u) * np.sin(v)
		z = self.z0+np.cos(v)
		ax.plot_wireframe(x, y, z, color="r",alpha=0.05)
		ax.scatter(self.x0, self.y0, self.z0, marker="o", color="black")
		ax.scatter(x1,y1,z1 , marker="o",color="red")
		ax.scatter(x2,y2,z2 , marker="o",color="red")
		ax.scatter(x3,y3,z3 , marker="o",color="red")
		ax.scatter(x4,y4,z4 , marker="o",color="red")
		ax.plot([x1, x2], [y1, y2], [z1, z2], color='g')
		ax.plot([x3, x4], [y3, y4], [z3, z4], color='g')
		ax.plot([x1, x3], [y1, y3], [z1, z3], color='g')
		ax.plot([x2, x4], [y2, y4], [z2, z4], color='g')
		ax.plot([self.x0, x1], [self.y0, y1], [self.z0, z1], color='g',alpha=0.5)
		ax.plot([self.x0, x2], [self.y0, y2], [self.z0, z2], color='g',alpha=0.5)
		ax.plot([self.x0, x3], [self.y0, y3], [self.z0, z3], color='g',alpha=0.5)
		ax.plot([self.x0, x4], [self.y0, y4], [self.z0, z4], color='g',alpha=0.5)


		
		return None
		
	def initialize_photon(self,theta,phi):
		photon = Photon(self.x0,self.y0,self.z0,theta,phi)
		return photon


class Scene:
	
	def __init__(self):
		L = 6
		self.x_min, self.x_max = -L, L
		self.y_min, self.y_max = -L, L
		self.z_min, self.z_max = -L, L

		ax.set_xlim3d(self.x_min,self.x_max)
		ax.set_ylim3d(self.y_min, self.y_max)
		ax.set_zlim3d(self.z_min, self.z_max)
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')

		return None
		
	def init_camera(self,x0,y0,z0,theta_min,theta_max,phi_min,phi_max):
		'''
		Initialize a Camera
		'''
		
		cam = Camera(x0,y0,z0,Ngrid=100)
		cam.direction(theta_min,theta_max,phi_min,phi_max)
		
		return cam


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scene1 = Scene()
#cam1 = scene1.init_camera(x0=0.0,y0=5.0,z0=5.0,theta_min=np.pi*0.5,theta_max=np.pi*0.25,phi_min=0.0,phi_max=np.pi*0.5)
cam1 = scene1.init_camera(x0=0.0,y0=0.0,z0=5.0,theta_min=np.pi*0.5,theta_max=np.pi*0.8,phi_min=0.0,phi_max=np.pi*0.5)
photon1 =cam1.initialize_photon(theta=np.pi/8.0,phi=np.pi*0.5)
photon1.step_forward(1)
photon1.step_forward(1)
print(photon1.return_position())
plt.show()


