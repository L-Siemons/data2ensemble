
'''
This is a stand alone script that will take the file containing the 
principle axis system and plot it on the structure. 

This script prepares a set of pymol commands that can be visualised in pymol

execute with python plot_principle_axis.py <pdb> <pas>
and then from the pymol command line
@principle_axis.pymol

note: this script requires cgo_arrow.py to be in the directory that pymol is in. 
'''

import MDAnalysis as md 
import sys
import numpy as np 

pdb = sys.argv[1]
pas = sys.argv[2]

print(f'PDB: {pdb}')

uni = md.Universe(pdb)
com = uni.select_atoms('all').center_of_mass()

f = open(pas)
axis = []
for i in f.readlines():
	axis.append([float(a) for a in i.split()[1:]])
f.close()

o = open('principle_axis.pymol', 'w')
o.write(f'load {pdb}; hide all ; show cartoon; run cgo_arrow.py\n')

colors = ['red', 'blue','green']
factors = [0.5,1, 1.5]
for i,j,k in zip(axis, colors, factors):
	i = np.array(i)
	axis_big = com+i*10*k
	o.write(f'cgo_arrow [{com[0]}, {com[1]}, {com[2]}], [{axis_big[0]}, {axis_big[1]}, {axis_big[2]}], color={j}\n')

o.write('zoom')
o.close()