import os

# region of interest for point spread function
ROI = 2

# star sensor pixel num
w = 1024
h = 1024

# star sensor foucs in metres
f = 58e-3

# field of view angle in degrees
FOVx = 20
FOVy = 20

# star catalogue path
catalogue_path = 'catalogue/Filtered_Below_5.6_SAO.csv'

# the standard deviation of white noise 
white_noise_std = 10

# type of noise, set by generate.py
type_noise = 'none'
# the standard deviation of mv noise 0.1Mv
mv_noise_std = 0.1
# the standard deviation of positional noise 2 pixels
pos_noise_std = 2
# the number of false stars
num_false_star = 2

# region radius in pixels
region_r = 300
# number of rings
num_ring = 30
# number of sectors
num_sector = 30
# least number of neighbor stars
num_neighbor_limit = 4

config_name = f"{os.path.basename(catalogue_path).rsplit('.', 1)[0]}_Size{w}x{h}_FOV{FOVx}x{FOVy}_Focus{f}_R{region_r}_Ring{num_ring}_Sector{num_sector}_Neighbor{num_neighbor_limit}"