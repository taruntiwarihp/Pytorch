import math

def angles_from_four_points(line1, line2, line3, line4):
    m1 = (line2[1] - line1[1]) / (line2[0] - line1[0] + 0.0000001)
    m2 = (line4[1] - line3[1]) / (line4[0] - line3[0] + 0.0000001)

    theta = math.atan(abs((m2-m1)/(1 + m1*m2)))
    final_theta = theta * 180 / math.pi

    return final_theta