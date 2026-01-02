import sys
import scipy
from math import cos, sin, pi, radians
from PIL import Image 
import numpy as np
import h5py
import matplotlib.pyplot as plt

SAMPLE_SIZE = int(sys.argv[1]) #I try to do eveything in angstroms
WHITE = (255,255,255)
BLACK = (0,0,0)
SPACING = 3.346
BOND_LENGTH = 1.418
LAYERS = 4
TOP = 4

DEG_TO_RAD = pi/180
TRANSLATION_LENGTH = BOND_LENGTH*sin(radians(60))/sin(radians(30))
A1_DEGREE = radians(150)
A2_DEGREE = radians(30)
A3_DEGREE = radians(60)
GRID_ROTATION = radians(62)
IMAGE_NUMBER = sys.argv[2]

#print([[BOND_LENGTH*cos(A1_DEGREE), BOND_LENGTH*sin(A1_DEGREE)], [BOND_LENGTH*cos(A2_DEGREE), BOND_LENGTH*sin(A2_DEGREE)]])
HEXAGONAL_BASIS = np.array([[TRANSLATION_LENGTH*cos(A1_DEGREE), TRANSLATION_LENGTH*sin(A1_DEGREE),0], [TRANSLATION_LENGTH*cos(A2_DEGREE), TRANSLATION_LENGTH*sin(A2_DEGREE),0], [BOND_LENGTH*cos(A3_DEGREE), BOND_LENGTH*sin(A3_DEGREE), SPACING]])
ROTATION = np.array([[cos(GRID_ROTATION), sin(GRID_ROTATION), 0], [-sin(GRID_ROTATION), cos(GRID_ROTATION), 0], [0, 0, 1]])
TRANSFORMATION_BASIS = np.matmul(HEXAGONAL_BASIS, ROTATION)
#print(TRANSFORMATION_BASIS)
INIT_STEPS = int(SAMPLE_SIZE/TRANSLATION_LENGTH)

#print(TRANSFORMATION_BASIS)

on_pixels = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE) ,dtype = bool)
#print(on_pixels.shape)
with Image.open(f"frames_small/out{IMAGE_NUMBER}.png") as img:
    
    for x in range(img.width):
        for y in range(img.height):
            pxl = img.getpixel((x,y))
            if (pxl != BLACK):
                on_pixels[x][y] = True

#print(on_pixels)
#on_pixels = np.array(on_pixels)

def is_on(point):
    #print(point[0])
    #print(on_pixels[int(point[0])][int(point[1])])
    return on_pixels[int(point[0])][int(point[1])]

#print(on_pixels)
#print(on_pixels.shape)

#plt.scatter(on_pixels[0:, 0], on_pixels[0:, 1], s=1)
#plt.show()

POINT1 = np.array([0,0,TOP], dtype = float)
POINT2 = np.array([BOND_LENGTH*cos(GRID_ROTATION),BOND_LENGTH*sin(GRID_ROTATION),TOP], dtype = float)
#POINT1[:2] += SAMPLE_SIZE/2
#POINT2[:2] += SAMPLE_SIZE/2 

#POINT2 = np.array([BOND_LENGTH,0,0])

graphite_plane1 = np.array([POINT1+TRANSFORMATION_BASIS[0]*i+TRANSFORMATION_BASIS[1]*j+TRANSFORMATION_BASIS[2]*k for i in range(SAMPLE_SIZE) for j in range(SAMPLE_SIZE) for k in range(LAYERS)])
graphite_plane2 = np.array([POINT2+TRANSFORMATION_BASIS[0]*i+TRANSFORMATION_BASIS[1]*j+TRANSFORMATION_BASIS[2]*k for i in range(SAMPLE_SIZE) for j in range(SAMPLE_SIZE) for k in range(LAYERS)])
#graphite_plane1 = np.array([POINT1+TRANSFORMATION_BASIS[0]*i+TRANSFORMATION_BASIS[1]*j for i in range(3) for j in range(3)])
#graphite_plane2 = np.array([POINT2+TRANSFORMATION_BASIS[0]*i+TRANSFORMATION_BASIS[1]*j for i in range(3) for j in range(3)])
graphite_plane = np.append(graphite_plane1, graphite_plane2, axis=0)

#print(graphite_plane)
#print(graphite_plane2)
#plt.figure(figsize=(9,9))
#plt.scatter(graphite_plane[0:, 0], graphite_plane[0:, 1], s=1)
#plt.scatter(POINT1[0], POINT1[1], s = 2)
#plt.scatter(POINT2[0], POINT2[1], s = 2)
#plt.xlim(left = 0, right = SAMPLE_SIZE)
#plt.ylim(bottom = 0, top = SAMPLE_SIZE)
#plt.show()

#calculate center of mass
com = np.average(graphite_plane, axis = 0, keepdims=True)
#print(com[0][2])
CENTER = np.array([SAMPLE_SIZE/2, SAMPLE_SIZE/2, com[0][2]])
#move = com - np.array([SAMPLE_SIZE/2])
#print(graphite_plane.shape)
#print(com)
com -= CENTER
graphite_plane -= com

#plt.scatter(graphite_plane[0:, 0], graphite_plane[0:, 1], s=1)

above = graphite_plane[:, 1] >= 0
below = graphite_plane[:, 1] < SAMPLE_SIZE-1
left = graphite_plane[:, 0] >= 0
right = graphite_plane[:, 0] < SAMPLE_SIZE-1
#print(graphite_plane)
layer1 = graphite_plane[:, 2] == (LAYERS-1)*SPACING + TOP
#layer2 = graphite_plane[:, 2] == (LAYERS-2)*SPACING + TOP
animation_layers = []
for i in range(0 , LAYERS-1):
    animation_layers.append(graphite_plane[:, 2] == SPACING*i + TOP)
    #print(animation_layers)
#print(animation_layers)
#print(np.unique(layer1))

keep = np.logical_and(np.logical_and(above,below), np.logical_and(left, right))

keep_layer_1 = np.logical_and(keep, layer1)
#keep_layer_2 = np.logical_and(keep, layer2)
animation_layers_rounded = []
for i in range (0, LAYERS-1):
    animation_layers[i] = np.logical_and(keep, animation_layers[i])
    animation_layers[i] = graphite_plane[animation_layers[i]]
    #print(np.unique(animation_layers[i][:, 2]))
    rounded = animation_layers[i].round()
    is_frame_on = [is_on(coor) for coor in rounded]
    animation_layers[i] = animation_layers[i][is_frame_on]

#print(above)
#print(keep)
graphite_layer_1 = graphite_plane[keep_layer_1]
#graphite_layer_2 = graphite_plane[keep_layer_2]
#print(graphite_layer_1)
#print(graphite_layer_2)
#print(type(graphite_layer_1))
#print(type(graphite_layer_2))
#everything = np.concatenate((graphite_layer_1, graphite_layer_2))
everything = graphite_layer_1
for i in range (0, LAYERS-1):
    everything = np.concatenate((everything, animation_layers[i]))
#print(everything.shape)
#print(everything)
#dic = {"atoms" : everything}
#scipy.io.savemat("test.mat", dic)
f = h5py.File(f"atom_frames/frame_{IMAGE_NUMBER}.h5", "w")
dset = f.create_dataset("atoms", data=everything)

#print(np.unique(graphite_layer_1[:, 2]))
#for coor in graphite_layer_2_closest:
    #print(coor)
    #print(is_on(coor))

#print(np.unique(keep_layer_1))
#print(graphite_plane)
#on_atoms = np.apply_along_axis(lambda vctr : np.matmul(HEXAGONAL_TRANSFORMATION, vctr), 1, on_pixels)
#print(on_atoms)

#plt.scatter(on_atoms[0:, 0], on_atoms[0:, 1], s=1)
#plt.show()
#plt.figure(figsize=(10,10))
#plt.scatter(graphite_layer_1[0:, 0], graphite_layer_1[0:, 1], s=1)
#for i in range(0, LAYERS-1):
#    plt.scatter(animation_layers[i][0:, 0], animation_layers[i][0:, 1], s=1)
#plt.scatter(graphite_layer_2[0:, 0], graphite_layer_2[0:, 1], s=1)
#plt.scatter(graphite_layer_2_closest[0:, 0], graphite_layer_2_closest[0:, 1], s=1)
#print(graphite_layer_2_on)
#plt.scatter(POINT1[0], POINT1[1], s = 2)
#plt.scatter(POINT2[0], POINT2[1], s = 2)
#plt.xlim(left = 0, right = SAMPLE_SIZE)
#plt.ylim(bottom = 0, top = SAMPLE_SIZE)
#plt.show()

