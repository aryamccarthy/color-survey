import colour

def munsell_to_rgb(munsell):
    xyY = colour.munsell_colour_to_xyY(munsell)
    XYZ = colour.xyY_to_XYZ(xyY)
    C = colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['C']
    RGB = colour.XYZ_to_sRGB(XYZ, C)

if __name__ == '__main__':
    print(munsell_to_rgb("E29"))