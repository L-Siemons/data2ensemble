
'''
This is a stand alone script that will take the file containing the 
principle axis system and plot it on the structure. 

This script prepares a set of pymol commands that can be visualised in pymol

execute with python plot_principle_axis.py <pdb> <pas> <angles>
and then from the pymol command line
@principle_axis.pymol

note: this script requires cgo_arrow.py to be in the directory that pymol is in. 
'''

import MDAnalysis as md 
import sys
import numpy as np 

pdb = sys.argv[1]
pas = sys.argv[2]
angs = sys.argv[3]

print(f'PDB: {pdb}')

uni = md.Universe(pdb)
com = uni.select_atoms('all').center_of_mass()

# do the angles
angles = open(angs)
data = {}
for i in angles.readlines():
	s = i.split()
	key = (s[0],s[1])
	data[key] = [float(a) for a in s[2:]]

angles.close()

# get the PAS
f = open(pas)
axis = []
for i in f.readlines():
	axis.append([float(a) for a in i.split()[1:]])
f.close()

#start writing he pymol file
o = open('principle_axis_for_all_atoms.pymol', 'w')
o.write(f'load {pdb}; hide all ; show cartoon; run cgo_arrow.py\n')

colors = ['red', 'blue','green']
factors = [0.5,1, 1.5]

for key in data:
	
	res, atom = key
	sele = uni.select_atoms(f'resid {res} and name {atom}')
	pos = sele.positions[0]

	for i,j,k in zip(axis, colors, factors):
		i = np.array(i)
		axis_big = pos+i*k
		o.write(f'pseudoatom object, pos=[{axis_big[0]}, {axis_big[1]}, {axis_big[2]}]\n')
		o.write(f'cgo_arrow [{pos[0]}, {pos[1]}, {pos[2]}], [{axis_big[0]}, {axis_big[1]}, {axis_big[2]}], color={j}\n')

o.write('zoom')
o.close()