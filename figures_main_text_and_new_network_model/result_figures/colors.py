
import numpy as np
colors = {}

def h2r(_hex):
    """
    Convert a hex string to an RGB-tuple.
    """
    if _hex.startswith('#'):
        l = _hex[1:]
    else:
        l = _hex
    return list(bytes.fromhex(l))

def r2h(rgb):
    """
    Convert an RGB-tuple to a hex string.
    """
    return '#%02x%02x%02x' % tuple(rgb)

def torgb(color):
    """
    Convert any color to an rgb tuple.
    """

    if isinstance(color,str):
        if len(color) in (6, 7):
            try:
                return h2r(color)
            except:
                pass
        try:
            return colors[color]
        except KeyError as e:
            raise ValueError("unknown color: '" + str(color) +"'")
    elif type(color) in (list, tuple, np.ndarray) and len(color) == 3:
        return h2r(color)
    else:
        raise ValueError("Don't know how to interpret color " + str(color))

def tohex(color):
    """
    Convert any color to its hex string.
    """

    if type(color) == str:
        if len(color) in (6, 7):
            try:
                h2r(color)
                return color
            except:
                pass
        try:
            return hex_colors[color]
        except KeyError as e:
            raise ValueError("unknown color: '" + color +"'")
    elif type(color) in (list, tuple, np.ndarray) and len(color) == 3:
        return r2h(color)
    else:
        raise ValueError("Don't know how to interpret color " + str(color))


def brighter(rgb,scl=2):
    """
    Make the color (rgb-tuple) a tad brighter.
    """
    _rgb = tuple([ int(np.sqrt(a/255.) * 255) for a in rgb ])
    _rgb = tuple([ (a/255.)**(1/scl) for a in rgb ])
    return _rgb


def darker(rgb,scl=1.5):
    """
    Make the color (rgb-tuple) a tad darker.
    """
    _rgb = tuple([ int((a/255.)**2 * 255) for a in rgb ])
    _rgb = tuple([ (a/255.)**scl for a in rgb ])
    return _rgb

import colorsys

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    rgb = [ i/255 for i in rgb ]
    h, l, s = colorsys.rgb_to_hls(*rgb)
    print(h,l,s)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l**(1/scale_l)), s = s)

def brighter(rgb,scl):
    return scale_lightness(rgb,scl)
