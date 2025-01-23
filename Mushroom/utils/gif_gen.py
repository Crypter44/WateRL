import os

import imageio

path = "/Mushroom/Plots/sac_simple_water/gif/"

files = os.listdir(path)

with imageio.get_writer(path + 'evaluation_after_entropy_tune.gif', mode='I', duration=750) as writer:
    for filename in files:
        image = imageio.imread(path + filename)
        writer.append_data(image)
