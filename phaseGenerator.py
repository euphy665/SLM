#################################################################
## This program can generate phase pattern for multiple useage ##
## including lens, grating, airy beam and block airy beam      ##
## all phase pattern with slm size and 0-1 range (0-2pi)       ## 
## Written by Ying in 2022.March                               ##
#################################################################

from cmath import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as LogNorm
import cv2
from scipy.fft import fft,fft2

class phaseGenerator():
    
    def __init__(self, wavelength = 795*10**-9, pixel_size = 8*10**-6 ,dim = (1080,1920)):
        self.phase = np.zeros(dim)
        self.k = 2*pi/wavelength
        self.dim = dim
        self.pixel_size = pixel_size

# This function is to generate the 2D array for x and y coordinator.
    def xyGenerator(self,theta = 0):
        x = np.arange(self.dim[1])-self.dim[1]/2
        y = np.arange(self.dim[0])-self.dim[0]/2
        y_2D = np.transpose(np.reshape(np.repeat(x,self.dim[0]),(self.dim[1],self.dim[0])))
        
        x_2D = np.reshape(np.repeat(y,self.dim[1]),(self.dim[0],self.dim[1]))

        theta = theta / 180 * np.pi
        y1 = x_2D * np.cos(theta) + y_2D * np.sin(theta)
        x1 = - x_2D * np.sin(theta) + y_2D * np.cos(theta)
        x = x1
        y = y1
        return x,y

# Generate lens pattern, fx and fy are the focal lens along x and y axis
# fx(fy) ==0 means the lens is cylindercal   
    def lens(self, fx, fy):
        x,y = self.xyGenerator()
        if fx == 0.0 and fy== 0.0:
            phase = np.zeros(self.dim)
        elif fx == 0.:
            phase = y**2/(2*fy)
        elif fy == 0.:
            phase = x**2/(2*fx)
        else:
            phase=x**2/(2*fx)+y**2/(2*fy)
        phase = phase*self.k*self.pixel_size**2
        return np.mod(phase/2/np.pi,1)

# This function generate blazed grating to move the image in focal plane by 'dx' or 'dy'
# with the focal length of lens located after SLM, the position move can be calculated
    def grating(self, dx, dy, f = 100*0.001):
        x,y = self.xyGenerator()
        theta_unit = 1 / 256
        grating_x = x*dx*theta_unit
        grating_y = y*dy*theta_unit
        return np.mod(grating_x + grating_y,1)

# This function calculate the discance between (x,y) and Ax+By+C=0
    def distance(self,x,y,A,B,C):
        return abs(A*x+B*y+C)/np.sqrt(A*A+B*B)

# This function generate phase pattern with blocked area, the area is rectangular
# The value of phase are 0 in blocked area, 1 in unbloced area
# means that this function should be multiple to the other functions instead of adding together
# Parameters: the center of the blocked rectangular position (xc, yc), the length of it (dx,dy) 
# and the rotated angle 'angle'
    def block(self, xc, yc, dx, dy, angle):
        phase = np.ones(self.dim)
        for i in np.arange(self.dim[0]):
            for j in np.arange(self.dim[1]):
                if angle == 0:
                    distance_y = abs(j-yc)
                    distance_x = abs(i-xc)
                elif angle == pi/2:
                    distance_x = abs(j-yc)
                    distance_y = abs(i-xc)                    
                else:
                    distance_y = self.distance(i,j,np.tan(angle),-1,-xc*np.tan(angle)+yc)
                    distance_x = self.distance(i,j,1,np.tan(angle),-xc-yc*np.tan(angle)) 
                if distance_y<(dy/2) and distance_x<(dx/2):
                    phase[i,j] = 0
        return np.mod(phase,2*pi) 

# This function is to generate cubic phase pattern for 1D airy beam, the essential parameter is 'scale'
# another parameter is the axis of the dimension, default value is x
# phase = (x*pixelsize*scale)^3
    def airyBeam1D(self, scale, axis = 'x',alpha=0, theta = 0):
        x,y = self.xyGenerator(theta = theta )
        if axis == 'x':       
           phase = (x**3)*(self.pixel_size*scale)**3
            #phase = (x**3+y**3)*(self.pixel_size*scale)**3
        else: 
            phase = (y**3)*(self.pixel_size*scale)**3
        return np.mod(phase/2/np.pi,1)

# This function is to generate cubic phase pattern for 2D airy beam, the essential parameter is 'scale'
# phase = (x*pixelsize*scale)^3
    def airyBeam2D(self, scale,alpha=0, theta = 0):
        x,y = self.xyGenerator(theta = theta )
        phase = (x**3+y**3)*(self.pixel_size*scale)**3
        return np.mod(phase/2/np.pi,1)

    def rotation(self, angle):
        x,y = self.xyGenerator()


    def imagePreview(self, phase, w0):
        x,y = self.xyGenerator()        
        Gaussian = np.exp(-(x**2+y**2)*self.pixel_size**2/w0**2)
        Image = fft2(Gaussian*np.exp(1j*phase))
        return Image

    

#    def saveImage(self, phase, path)

if __name__ == '__main__':

## User guide, first generate the class
    test = phaseGenerator()

## Generate lens, grating and block, 1D airy beam, 2D airy beam
    # phase1 = test.lens(1,0)
    # phase2 = test.grating(1,0.5,0.1)
    # phase3 = test.block(500,500,400,200,pi/2)
    phase6 = test.airyBeam2D(500,theta = 0)
    # phase4 = test.airyBeam1D(500,axis= 'x')
    # phase5 = test.airyBeam1D(500,axis = 'y')
    fig, axes = plt.subplots(2,2)  
    # axes[0,0].pcolor(phase6)
    axes[0,1].pcolor(phase6)
    # axes[1,0].pcolor(np.transpose(phase6))
    # axes[1,1].pcolor(phase4)
    plt.show()
## Save to file
    cv2.imwrite('Airytest_2.jpg',np.transpose(phase4/2/pi*255))
    cv2.imwrite('Airytest_3.jpg',np.transpose(phase5/2/pi*255))