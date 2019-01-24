"""
data processing plugin for camera IOC

It defines the following entry point:
* init
* calc
* free
"""
import logging
import os.path
import math

import numpy

logging.basicConfig(format='%(asctime)-15s %(funcName)s %(message)s', level=logging.INFO)

def init(size_x, size_y, size_z, device):
    """
    init function to be called after image information changes
    """
    logging.info('size_x = %s' % size_x)
    logging.info('size_y = %s' % size_y) 
    logging.info('device = %s' % device)


def calc(sizedata, data):
    """
    calc function to be called each time new image arrives
    """
    # image is readonly, make a copy if you want to change
    image = data['waveinput'][0]
    logging.info('image size = %s' % str(image.shape))

    # process the image and fill in the results
    results = {}
    results['min'] = image.min()
    results['max'] = image.max()
    results['mean'] = image.mean()
    results['profile_x'] = image.sum(axis=0)
    results['profile_y'] = image.sum(axis=1)

    x0, y0, sigmax, sigmay, orientation = momentum(image)
    results['center_x'] = x0
    results['center_y'] = y0
    results['sigma_x'] = sigmax
    results['sigma_y'] = sigmay
    results['orientation'] = orientation

    logging.info(results)

    return results


def free(device):
    """
    free function to be called before image information changes
    """
    logging.info('device = %s' % device)


#
# Processing routines
#
def momentum(image, x=None, y=None):
    """
    Computes spatial moments of an image.

    Translated from Francois Richard's ImageJ plugin Moment_Calculator
    https://imagej.nih.gov/ij/plugins/download/Moment_Calculator.java

    Note that spatial moments are a very simple and powerful way to describe the
    spatial distribution of values, provided they have a sufficiently strong
    central tendency, that is, a tendency to cluster around some particular
    value. This implies that "background" pixel values are small(e.g. zones where
    the quantity of interest, such as concentration, is zero).
    Conversely, zones of high concentration (density, etc.) should also have a
    high pixel values. This can lead to meaningless results, for example, in the
    case of uncalibrated images, where (white) background pixels are equal to 255
    (for an 8-bit greyscale image).
   
    ** Interpretation of spatial moments **
             
      * order 0  = TOTAL MASS [units: concentration, density, etc.]
      * order 1  = location of CENTRE OF MASS in x and y from 0,0 [units: L]
      * order 2  = VARIANCE (spreading) around centroid in x and y [units: L^2]
      * order 3  = coeff. of SKEWNESS (symmetry) in x and y [units: n/a]
                               -->  =0  : SYMMETRIC distribution
                               -->  <0  : Distribution asymmetric to the LEFT
                                          (tail extends left of centre of mass)
                               -->  >0  : Distribution asymmetric to the RIGHT
                                         (tail extends right of centre of mass)
      * order 4  = KURTOSIS (flatness) in x and y [units: n/a]
                               --> =0   : Gaussian (NORMAL) distribution
                               --> <0   : Distribution FLATTER than normal
                               --> >0   : Distribution MORE PEAKED than normal
                               --> <-1.2: BIMODAL (or multimodal) distribution
                               
    ** Parameters derived from 2nd moments ** (from Awcock (1995) "Applied Image Processing")
   
      * ELONGATION (ECCENTRICITY) = Ratio of longest to shortest distance vectors
                                    from the object's centroid to its boundaries
      * ORIENTATION = For elongated objects, describes the orientation (in degrees)
                      of the "long" direction with respect to horizontal (x axis)
    """
    if x is None:
        x = numpy.arange(image.shape[1])
    if y is None:
        y = numpy.arange(image.shape[0])

    m00 = image.sum()
    m10 = numpy.sum(numpy.dot(image, x))
    m01 = numpy.sum(numpy.dot(y, image))

    x0 = m10 / m00
    y0 = m01 / m00

    m20 = numpy.dot(image, (x-x0)**2).sum()
    m02 = numpy.dot((y-y0)**2, image).sum()
    m11 = numpy.dot(numpy.dot(y-y0, image), x-x0).sum()

    xxVar = m20 / m00
    yyVar = m02 / m00;
    xyVar = m11 / m00;

    sigmax = math.sqrt(xxVar)
    sigmay = math.sqrt(yyVar)

    # Compute Orientation and Eccentricity
    # source: Awcock, G.J., 1995, "Applied Image Processing", pp. 162-165
    orientation = 0.5*math.atan2((2.0*m11),(m20-m02))
    orientation = orientation*180./math.pi #convert from radians to degrees
    eccentricity = ((m20 - m02)**2 + 4 * m11**2) / (m20+m02)**2

    return x0, y0, sigmax, sigmay, orientation
