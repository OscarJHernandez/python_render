import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import objects.ball as ball
import objects.wall as wall
import light.photon as photon
import light.point_source as point_source
import utils as utils




class Camera:
	
	def __init__(self,x0,y0,z0,radius,Ngrid):
		self.x0 = x0
		self.y0 = y0
		self.z0 = z0
		self.radius = radius
		self.image = np.zeros((Ngrid,Ngrid))
		return None
		
	def convert_spherical_to_cartesian(self,theta,phi):

		print("theta: ", theta, "phi: ", phi)
		x = self.x0+self.radius*np.sin(theta)*np.cos(phi)
		y = self.y0+self.radius*np.sin(theta)*np.sin(phi)
		z = self.z0+self.radius*np.cos(theta)

		p = np.zeros(3)
		p[0],p[1],p[2] = x,y,z

		print("x y z: ", x,y,z)
		print("")

		return p

	def detection(self,p1):
		'''
		Updates the Camera grid if it detected a photon

		:param p1:
		:return:
		'''

		detection = False
		tol = 1e-2

		x1,y1,z1 = p1
		normal = self.normal
		d_val = self.d
		p0 = self.p_upper_right


		# Compute the distance from the photon point to the plane
		distance = abs(normal.dot(p0-p1))


		if(distance< tol and self.x_min <= x1<= self.x_max and self.y_min <= y1<= self.y_max and self.z_min <= z1<= self.z_max):
			detection = True

		return detection

		

	def update_image(self,photon):
		'''
		update the image that the camera has

		:param photon:
		:return:
		'''





		return None


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

		# Save the values
		self.p_upper_left = p_upper_left
		self.p_upper_right = p_upper_right
		self.p_lower_left = p_lower_left
		self.p_lower_right = p_lower_right

		x1,y1,z1 = p_lower_left
		x2, y2, z2 = p_lower_right

		x3,y3,z3 = p_upper_left
		x4, y4, z4 = p_upper_right

		# The normal
		v1 = p_upper_left- p_lower_left
		v2 = p_lower_right - p_lower_left
		n_surface = np.cross(v1,v2)
		n_surface = n_surface/np.linalg.norm(n_surface)

		self.normal = n_surface

		# ax+by+c*z +d = 0
		d  = -p_upper_right.dot(n_surface)

		self.d = d

		# create x,y
		x_min = np.min([x1,x2,x3,x4])
		x_max = np.max([x1,x2,x3,x4])
		y_min = np.min([y1,y2,y3,y4])
		y_max = np.max([y1,y2,y3,y4])
		z_min = np.min([z1,z2,z3,z4])
		z_max = np.max([z1,z2,z3,z4])
		xx, yy = np.meshgrid(np.linspace(x_min,x_max,100) , np.linspace(y_min,y_max,100))

		self.x_min = x_min
		self.x_max = x_max
		self.y_min = y_min
		self.y_max = y_max
		self.z_min = z_min
		self.z_max = z_max

		# calculate corresponding z
		z = -(n_surface[0]*xx+n_surface[1]*yy +d)/n_surface[2]
		z[z>z_max] = np.nan
		z[z<z_min] = np.nan

		print("Z: ", z.shape)
		print("Z", z[0,0], xx[0,0],yy[0,0])

		ax.plot_surface(xx, yy, z, alpha=1.0)
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
		
	def initialize_photon(self,theta,phi,wavelength=600):
		return photon.Photon(self.x0,self.y0,self.z0,theta,phi,wavelength=wavelength)

class Scene:
	
	def __init__(self,L=6):

		self.x_min, self.x_max = -L, L
		self.y_min, self.y_max = -L, L
		self.z_min, self.z_max = -L, L
		self.objects = []

		ax.set_xlim3d(self.x_min,self.x_max)
		ax.set_ylim3d(self.y_min, self.y_max)
		ax.set_zlim3d(self.z_min, self.z_max)
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')


		self.cams = []

		return None
		
	def init_camera(self,x0,y0,z0,theta_min,theta_max,phi_min,phi_max,radius):
		'''
		Initialize a Camera
		'''
		
		cam = Camera(x0,y0,z0,radius=radius,Ngrid=100)
		cam.direction(theta_min,theta_max,phi_min,phi_max)

		self.cams.append(cam)
		
		return None

	def init_ball(self,x0=0.0,y0=0.0,z0=0.0,radius=1.0,ax=None):
		'''

		:param x0:
		:param y0:
		:param z0:
		:param radius:
		:return:
		'''

		b1 = ball.Ball(x0,y0,z0,radius,ax=ax)
		self.objects.append(b1)

		# walls

		return None

	def init_wall(self,x0=0.0,y0=0.0,z0=0.0,Lx=0.0,Ly=0.0,Lz=0.0,ax=None):
		w1 = wall.Wall(x0,y0,z0,Lx,Ly,Lz,ax=ax)
		self.objects.append(w1)
		return None

	def check_photon(self,photon):
		'''
		Check to see if the photon is in the scene

		:param photon:
		:return:
		'''

		in_scene = False

		xp,yp,zp = photon.return_position()

		if( self.x_min<xp< self.x_max and self.y_min < yp < self.y_max and self.z_min < zp < self.z_max):
			in_scene = True

		return in_scene

	def init_light(self,x0,y0,z0,T=220,ax=None):
		'''
		Initialize the light source

		:param x0:
		:param y0:
		:param z0:
		:param T:
		:param ax:
		:return:
		'''

		light = point_source.PointSource(x0,y0,z0,T=220,ax=ax)

		# Generate many photons in random directions
		photons = light.generate_photons(N_photons=20000)


		# Time evolve all of the photons, loop over objects in the scene
		# depending on the type of object, resolve the collision
		for t in range(0,800):

			for k in range(len(photons)):

				in_scene = self.check_photon(photons[k])

				# Only evlolve in-scence photons
				if (in_scene == True):
					photons[k].step_forward(0.05)

				photon_position = photons[k].return_position()
				xp,yp,zp = photon_position

				# Loop over all the objects in the scene to check for photon-object collisions
				for i in range(len(self.objects)):
					obj = self.objects[i]

					if(obj.check_collision(photon_position)==True ):
						photons[k] = obj.resolve_collision(photons[k])
						ax.scatter(xp,yp,zp, marker=".", color=utils.wavelength_to_rgb(photons[k].wavelength,gamma=1.0),alpha=1.0)

				# Loop over all the cameras to update their images
				for i in range(len(self.cams)):
					cam = self.cams[i]

					if(cam.detection(photon_position)== True and photons[k].reflected== True):
						ax.scatter(xp,yp,zp, marker="o", color=utils.wavelength_to_rgb(photons[k].wavelength,gamma=1.0),alpha=0.3)
						print('Photon detected!')


		#for k in range(len(photons)):

		#	photons[k].plot_photon(ax)

			#if(photons[k].reflected==True):
			#	photons[k].plot_photon(ax)

		return light


if __name__ == '__main__':

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	scene1 = Scene(L=6)
	#cam1 = scene1.init_camera(x0=0.0,y0=5.0,z0=5.0,theta_min=np.pi*0.5,theta_max=np.pi*0.25,phi_min=0.0,phi_max=np.pi*0.5)
	scene1.init_camera(x0=-5.0,y0=-5.0,z0=2.0,theta_min=np.pi*0.25,theta_max=np.pi*0.75,phi_min=0.0,phi_max=np.pi*0.5,radius=4)
	#scene1.init_camera(x0=-4.0, y0=-4.0, z0=1.0, theta_min=np.pi * 0.25, theta_max=np.pi * 0.75, phi_min=0.0,phi_max=np.pi * 0.5, radius=3)

	cam1 = scene1.cams[0]
	photon1 =cam1.initialize_photon(theta=np.pi/8.0,phi=np.pi*0.5,wavelength=600)
	scene1.init_ball(ax=None,radius=2)
	#scene1.init_wall(ax=None, z0=6, Lx=5.0, Ly=5.0, Lz=0.0)
	#
	scene1.init_wall(ax=None,z0=-5,Lx=5.0,Ly=5.0,Lz=0.0)
	scene1.init_light(x0=-2.0,y0=-2.0,z0=3.0,T=220,ax=ax)
	#scene1.init_light(x0=-3.0, y0=-3.0, z0=5.0, T=220, ax=ax)
	plt.show()


