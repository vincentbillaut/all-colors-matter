import numpy as np


################################################################################
# util functions for translating RGB to YUV and vice versa inspired from:
# https://gist.github.com/Quasimondo/c3590226c924a06b276d606f4f189639
# and a little bit modified

# input is a RGB numpy array with shape (height,width,3), can be uint,int,float
# or double, values expected in the range 0..255
# output is a double YUV numpy array with shape (height,width,3), values in the
# ranges [0,1], [-0.436,.436], [-.615,.615]
def RGB_to_YUV(rgb):
    m = np.array([[0.29900, -0.14713333333333333333, 0.615],
                  [0.58700, -0.28886666666666666666, -0.51498888888888888888888],
                  [0.11400, 0.436, -0.10001111111111105]])

    yuv = np.dot(rgb, m)
    yuv /= 255
    assert (rgb.shape == yuv.shape)
    return yuv


# input is an YUV numpy array with shape (height,width,3) of floats expected in the range
#  [0,1],[−0,436 ; 0,436], [−0,615 ; 0,615]
# output is a double RGB numpy array with shape (height,width,3), values in the range 0..255
def YUV_to_RGB(yuv,correction = True):
    m = np.array([[ 1.0,  1.0, 1.0],
                   [-8.7249120380940816e-06, -3.9464658045574741e-01, 2.0321065918966941],
                   [1.1398353110156241, -5.8059458027813926e-01, -1.5257635121515506e-05]])
    rgb = np.dot(yuv, m)
    rgb*=255.
    if correction:
        rgb[rgb>255]=255
        rgb[rgb<0]=0
    assert (rgb.shape == yuv.shape)
    return rgb


################################################################################

def variance_UV_channels(yuv):
    return np.var(yuv[:, :, 1:])
