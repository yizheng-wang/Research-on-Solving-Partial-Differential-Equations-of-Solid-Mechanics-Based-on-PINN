# DCM by gaussian integration

import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
import matplotlib as mpl
import numpy as np
from pyevtk.hl import pointsToVTK
from torch.optim.lr_scheduler import MultiStepLR
from matplotlib import cm

    

# =============================================================================
# Loading gaussian points including boundary and domain
# =============================================================================
file = open("rectangle.inp")
lines = file.readlines()
file.close()
# the first occurrence of #Node is the coordinate of the nodes, we storage themin the array
DIFF_POINTS = lines.index("*Node\n")  # for gauss integration points
DIFF_POINTS_end = lines.index("*Element, type=CPS8R\n")  # the end mark of gauss points

node_coordinate = lines[DIFF_POINTS+1 : DIFF_POINTS_end]
node_coordinate = [i.replace(",", "").strip().split()[1:] for i in node_coordinate]
node_coordinate = [list( map(float,i) ) for i in node_coordinate]
node_coordinate = np.array(node_coordinate)


file = open("rectangle.rpt")
lines = file.readlines()
file.close()
# the first occurrence of #Node is the coordinate of the nodes, we storage themin the array
DIFF_POINTS = lines.index("---------------------------------------------------------------------------------\n")  # for gauss integration points
DIFF_POINTS_end = lines[DIFF_POINTS:].index("\n")  # the end mark of gauss points
node_stress = lines[DIFF_POINTS+1 : DIFF_POINTS + DIFF_POINTS_end]
node_stress = [i.strip().split()[1:] for i in node_stress]
node_stress = [list( map(float,i) ) for i in node_stress]
node_stress = np.array(node_stress)

np.save('node_coordinate_abaqus_rectangle', node_coordinate)
np.save('node_stress_abaqus_rectangle', node_stress)