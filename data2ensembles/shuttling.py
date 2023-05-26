
import numpy as np
import matplotlib.pyplot as plt 
import pkg_resources 
import sys
import scipy

class ShuttleTrajectory():

	def __init__(self, fields_file, magnetic_fields_trajectory_file='default', distance_incrementing=0.1):

		if magnetic_fields_trajectory_file == 'default':
			magnetic_fields_trajectory_file = pkg_resources.resource_filename('data2ensembles', 'dat/field_dist.txt')
		
		self.fields_distances_total = np.genfromtxt(magnetic_fields_trajectory_file, delimiter="\t")
		# we dont need more than milli tesla resolution ... I dont think ...

		added_fields = []
		self.fields_distances = []
		self.sampled_fields = []
		self.experiment_info = {}

		# this section assumes that the fields file is written in assending order
		distance_to_field = scipy.interpolate.interp1d(
			self.fields_distances_total.T[1], 
			self.fields_distances_total.T[0])

		# interpolate the curve 
		field_max = max(self.fields_distances_total.T[1])
		field_min = min(self.fields_distances_total.T[1])
		print(field_min	, field_max, distance_incrementing)
		fields = np.arange(field_min, field_max, distance_incrementing)
		print(fields)
		print(fields.shape)
		distances = distance_to_field(fields)

		self.fields_distances = np.array([distances, fields]).T
		self.fields_distances_dict = {}

		for i in self.fields_distances_total:
			self.fields_distances_dict[i[1]] = i[0]

		fields_data = np.genfromtxt(fields_file, delimiter="\t", dtype=None)
		for i in fields_data:
			field = float(i[1])
			self.experiment_info[field] = {}
			self.experiment_info[field]['high_field'] = float(i[0])
			self.experiment_info[field]['travel_time'] = float(i[2])
			self.experiment_info[field]['stabalisation_delay'] = float(i[3])
			self.experiment_info[field]['delays'] = [float(a) for a in i[4:]]
			self.sampled_fields.append(field)

		self.sampled_fields = np.array(self.sampled_fields)

	def construct_single_trajectory(self, field, all_dists, all_fields):

		# print('==start', field)
		# get the travel time set by the user 
		travel_time = self.experiment_info[field]['travel_time']
		half_time = travel_time/2
		
		# the distance that corresponds to the field
		total_distance = self.fields_distances_dict[field]
		half_distance = total_distance/2

		# print('travel time', travel_time)
		# print('total distance', total_distance)
		# print('half distance', half_distance)

		acceleration = half_distance*2/(half_time**2)
		decceleration = -1*acceleration
		max_velocity = acceleration*half_time

		#the distances where we are accelerating
		acc_selector = (all_dists<half_distance)
		acc_distances = all_dists[acc_selector]
		acc_time = np.sqrt(2*acc_distances/acceleration)

		# print('acc_distances', acc_distances)
		# print('acc', acc_time)

		#the distances where we are decellerating
		deccel_selector = (all_dists >= half_distance) &  (all_dists < total_distance)
		decc_distances = all_dists[deccel_selector] - half_distance

		#to get the decelleration times we will use the numpy solver 
		a = 0.5*decceleration
		b = max_velocity
		c = -1 * decc_distances

		decc_times = np.real(np.array([np.roots([a,b,ci])[1] for ci in c])) + half_time 
		decc_distances = decc_distances+half_distance

		# print('decc_distances', decc_distances)
		# print('decc_times', decc_times)

		# plots time vs distance 
		# plt.plot(acc_time, acc_distances)

		# plt.plot(decc_times, decc_distances)
		# plt.xlabel('time (s)')
		# plt.ylabel('distance (m)')
		# plt.show()

		total_time = np.concatenate([acc_time, decc_times])
		fields_selector = (all_dists<total_distance)
		# print(fields_selector.shape)
		# print(all_fields.shape)

		selected_fields = all_fields[fields_selector]
		# # plots time vs distance 
		# plt.plot(total_time, selected_fields)
		# plt.xlabel('time (s)')
		# plt.ylabel('field (T)')
		# plt.show()		

		#time spent at each field going forwards
		field_centers = (selected_fields[1:] + selected_fields[:-1]) / 2
		time_taken = total_time[1:] - total_time[:-1]

		# #plot time at each field
		# plt.plot(time_taken, field_centers)
		# plt.scatter(time_taken, field_centers)
		# plt.xlabel('time (s)')
		# plt.ylabel('field (T)')
		# plt.yscale('log')
		# plt.show()		


		return field_centers, time_taken

	def construct_all_trajectories(self, initial_velocity=0.):

		self.trajectory_time_at_fields = {}

		all_dists_master = self.fields_distances.T[0]
		all_fields_master = self.fields_distances.T[1]

		# there has been some confusion here 
		# I have used self.fields_distances instead of all_dists and all_fields
		# so I need to re-write this using these instead. 
		
		# print(all_fields.shape)
		# print(all_dists.shape)

		for field in self.sampled_fields: 
			# print(field)
			forwards = self.construct_single_trajectory(field, all_dists_master, all_fields_master)
			forwards_field_center, forwards_time_taken = forwards

			reverse_selector = (all_fields_master>=field)
			all_fields = np.flip(all_fields_master[reverse_selector])
			all_dists = np.flip(self.fields_distances_dict[field] - all_dists_master[reverse_selector])

			# since the path is symetrical we can just do the reverse of the forwards path
			backwards_field_center = np.flip(forwards_field_center)
			backwards_time_taken = np.flip(forwards_time_taken)

			# print('back')
			# backwards = self.construct_single_trajectory(field, all_dists, all_fields)
			# backwards_field_center, backwards_time_taken = backwards

			self.trajectory_time_at_fields[field] = [forwards_field_center, forwards_time_taken, 
													backwards_field_center, backwards_time_taken]

