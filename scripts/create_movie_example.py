#!/usr/bin/env python
"""
movie.py

Kelvin Chu, Department of Physics, University of Vermont
http://www.uvm.edu/~kchu/

A simple script to draw a pendulum oscillating and record it to a
movie. This works on MacOSX but should be easily extensible to
Linux/Unix. It does not work on Microsoft Windows, because the
commands module (shell command line) does nothing, and some system
tools are not available in the form used here.

Additional details (and the resulting animated gif and QuickTime
movie) can be found at:

   http://www.uvm.edu/~kchu/?Page=VPython/index.html

Strategy:
--------

1.  Use VPython to make your animation.

2.  At every step in the animation (i.e. every re-rendering of
    objects) take a screenshot (/usr/sbin/screencapture) and save the
    file.

3.  Use Imagemagick to crop out the image of the VPython animation and
    turn it into an animated gif file or movie file (using QuickTime).

    - I manually position the VPython window in the top left-hand
      corner of the screen, and then let the animation run.
      
    - For the default VPython window settings, one can crop with a
      setting of 430x365+0+0.

Requirements:
------------
0. An Apple Macintosh running MacOSX (currently Panther or Jaguar)
1. Fink (fink.sf.net)
2. Imagmagick (install via fink: 'fink install imagemagick')
3. VPython (www.vpython.org)
4. QuickTime Player

Movie comments are denoted by ** movie **

Some physics: The actual differential equation,

d(dtheta/dt)/dt = -(g/L)sin(theta) 

needs to be integrated.


"""

from visual import *                    # vpython module
from commands import *                  # shell/command line module; Unix only


def GetScreenShot(FrameNumber):         # Take a screenshot and write
                                        # it to a numbered file.
    tmp = getoutput('/usr/sbin/screencapture pendulum.%03d.pdf' % FrameNumber)
    print 'Frame: %d' % (FrameNumber)
    return

def GetRVector(Theta):                  # Calculate the position of
                                        # the mass based upon \theta
                                        # measured from the vertical.
    
    return vector(sin(Theta),-cos(Theta),0)

def CalculatePosition(Theta,ThetaDot,dt,l):
    
                                        # Integrate the differential
                                        # equation to get the new
                                        # \theta and \dot{\theta}
                                        # based upon previous position.
    g = 9.81

    NewThetaDot = -g/l*sin(Theta)*dt + ThetaDot
    NewTheta = NewThetaDot*dt + Theta 
    return NewTheta,NewThetaDot


# Some initial conditions and variable settings.
#

g = 9.81                                # Acceleration from gravity

Length = 1.0                            # Length of the pendulum bob.

Theta = pi/2                            # Initial angle, measured
                                        # relative to the vertical.

ThetaDot = 0                            # Initial velocity.

Deltat = .01                            # Time step (.01 seconds)

FrameNumber = 0                         # Counter for frames.




R = GetRVector(Theta)                   

# Draw the ground as something vaguely interesting.
box(pos = vector(-Length/2,-.5 - Length,-Length/2),height = 0.1, 
             length = Length,width = 1.0,color=color.white)
box(pos = vector( Length/2,-.5 - Length, Length/2),height = 0.1, 
             length = Length,width = 1.0,color=color.white)
box(pos = vector(-Length/2,-.5 - Length, Length/2),height = 0.1, 
             length = Length,width = 1.0,color=color.blue)
box(pos = vector( Length/2,-.5 - Length,-Length/2),height = 0.1, 
             length = Length,width = 1.0,color=color.blue)
# Anchor for the pendulum.
sphere(radius = 0.02,color=color.red,pos=vector(0,0,0))

# Pendulum itself
Dot = sphere(radius=0.07,color=color.yellow,pos=R)
Shaft = cylinder(radius = 0.02, color=color.white, pos=vector(0,0,0),
                 axis=R)

scene.autoscale = 0                     # Need to turn off autoscaling
                                        # to prevent insanity.

# **movie** By positioning the VPython window in the top left hand corner, we
# know its position on the screen and then can correctly crop it out
# of the screenshot afterwards.
#
print 'This program works only on MacOSX but could be extended to Unix/Linux.'
print 'This program does NOT work on Microsoft Windows.'
raw_input('Position VPython window at top left hand corner. Then hit enter in this window.')
#


while FrameNumber < 235:                # Period of oscillation is ~2.35s.
    Dot.pos = R
    Shaft.axis = R
    Theta,ThetaDot = CalculatePosition(Theta,ThetaDot,Deltat,Length)
    R = GetRVector(Theta)

# **movie** Use the screencapture utility to get a screenshot.  
    GetScreenShot(FrameNumber)
    FrameNumber = FrameNumber + 1
#

# **movie** Use ImageMagick to process the files, cropping out the
# unwanted portions of the screenshot and converting to an animated
# gif.  
print 'Converting files.'

# The arguments to convert are:
# -delay 01     : space a 1/100th of a second delay in between frames.
# -loop 0       : animated gif will repeat ad infinitum
# -crop AxB+C+D : crop the image with aspect AxB, offsets at C and D.
#                 You will have to experiment with these numbers if
#                 you change the VPython window size.
#
tmp = getoutput('/sw/bin/convert -delay 01 -loop 0 -crop 430x365+0+0 pendulum.*.pdf animated.pendulum.gif')
print 'File written to "animate.pendulum.gif". If you like it, you can delete the PDFs'
print 'Done' 
# 