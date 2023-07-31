
import numpy as np
import matplotlib.pyplot as plt 
import pkg_resources 
import sys
import scipy

class ShuttleTrajectory():
    """This class impliments model free calculations

    Parameters
    ----------
    fields_file : str 
        A file containing information regarding the relaxometry experiment. The lines should be as follows
        #high field, low field, travel time, stablisation delay, delays used the relaxation experiments

    magnetic_fields_trajectory_file : str 
        A file path with the distance from the magnetic center and the magnetic field. If left as ''default'
        the file corresponds to the shuttle system on the 600MHz at ENS Paris. This file should be written in 
        acending order

    field_increment : float 
        when simulating the trajectory of the shuttle we do it in discrete steps where each step is 
        seperated by a gap of a Tesla. The field incriment is this gap. The default value is 0.1T

    Attributes
    ----------
    distance_field_measured : ndarray 
        2D of distance vs field. The shape should be (N, 2) depending on how many measurements were made

    fields_distances : ndarray 
        this is an interpolated version of distance_field_measured where the fields are sampled at the 
        regular increments determined by field_increment.

    fields_distances_dict : dict 
        A dictionary where the key is the measured field and the entry is the distance

    sampled_fields : list 
        A list of the low fields that were sampled

    experiment_info : dict
        a dictionary contaning the following keys:
            high_field : float 
            travel_time : float 
            stabalisation_delay : float 
            delays : list

    Methods
    -------
    field_to_distance(a)
        This is a scipy 1D interpolator to convert the distance to the 
        field strngth in the magnet

    construct_single_trajectory(field, all_dists, all_fields)
        This method constructs the trajectory and returns how long you are at each field during shuttling



    """

    def __init__(self, fields_file, magnetic_fields_trajectory_file='default', field_increment=0.5):

        # load the magnetic field file v distance
        if magnetic_fields_trajectory_file == 'default':
            magnetic_fields_trajectory_file = pkg_resources.resource_filename('data2ensembles', 'dat/field_dist.txt')
        
        self.distance_field_measured = np.genfromtxt(magnetic_fields_trajectory_file, delimiter="\t")

        # this section assumes that the fields file is written in assending order
        # print('fields', self.distance_field_measured.T[1])
        # print('distance', self.distance_field_measured.T[0])
        
        self.field_to_distance = scipy.interpolate.interp1d(
            self.distance_field_measured.T[1], 
            self.distance_field_measured.T[0])

        self.distance_to_field = scipy.interpolate.interp1d(
            self.distance_field_measured.T[0], 
            self.distance_field_measured.T[1])
        # fields = np.linspace(1, 14, 100)
        # test = self.field_to_distance(fields)
        # plt.plot(fields, test)
        # plt.show()

        # interpolate the curve 
        field_max = max(self.distance_field_measured.T[1])
        field_min = min(self.distance_field_measured.T[1])

        fields = np.arange(field_min, field_max, field_increment)
        distances = self.field_to_distance(fields)

        # this is the trajectory with a given increment in tesla
        # fabien said that I could have the fields on a log scale
        self.fields_distances = np.array([distances, fields]).T

        self.fields_distances_dict = {}
        for i in self.distance_field_measured:
            self.fields_distances_dict[i[1]] = i[0]


        fields_data = np.genfromtxt(fields_file, delimiter="\t", dtype=None)

        self.sampled_fields = []
        self.experiment_info = {}
        for i in fields_data:
            field = float(i[1])
            self.experiment_info[field] = {}
            self.experiment_info[field]['high_field'] = float(i[0])
            self.experiment_info[field]['travel_time'] = float(i[2])
            self.experiment_info[field]['stabalisation_delay'] = float(i[3])
            self.experiment_info[field]['delays'] = np.array([float(a) for a in i[4:]])
            self.sampled_fields.append(field)

        self.sampled_fields = np.array(self.sampled_fields)

    def construct_single_trajectory(self, field, points=20):
        '''
        This funcion calculates the time spent at each field during shuttling. 
        Here we assume that shuttling occurs using a linear acceleration to the 
        mid point and then a linear decelleration afterwards where the magnitude 
        of the acceleration and decceleration are the same. These are calculated
        using the travel time. 


        Parameters
        ----------
        field : float 
            the low field

        all_dists : list, ndarray
            an array of the distances for each field 

        all_fields  : list, ndarray
            an array of all the fields in the shuttling trajectory

        Returns
        -------
        field_centers : ndarry 
            the field centers in the trajectory 

        time_taken : ndarry 
            the time spent at each field center. 
        '''

        #these are the points we travel through
        high_field_possition = self.experiment_info[field]['high_field']
        max_field_in_interpolator = np.max(self.distance_field_measured.T[1])

        fields_trajectory = np.geomspace(max_field_in_interpolator, field, points)
        distances = self.field_to_distance(fields_trajectory)

        # for debugging!
        # print(field)
        # print(max_field_in_interpolator)
        # print('F', fields_trajectory)
        # print('Dist', distances)
        # print(self.fields_distances_dict[field])

        # get the travel time set by the user
        travel_time = self.experiment_info[field]['travel_time']
        half_time = travel_time/2

        
        # the distance that corresponds to the field
        # here I could use the know distance but this might 
        # cause some slightly funny behaviour if there is a small error in the interpolator
        total_distance = np.max(distances)
        half_distance = total_distance/2

        # now we add the zero point for the highfield and also the point where we change acceleration
        midpoint_field = self.distance_to_field(half_distance)

        fields_trajectory = np.concatenate([[high_field_possition, midpoint_field], fields_trajectory])
        distances = np.concatenate([[0, half_distance], distances])
        indices = np.argsort(distances)
        distances = distances[indices]
        fields_trajectory = fields_trajectory[indices]


        # calculate the acceleration
        acceleration = half_distance*2/(half_time**2)
        decceleration = -1*acceleration
        max_velocity = acceleration*half_time

        #the distances where we are accelerating, include the midpoint in the acceleration
        acc_distances = distances[distances <= half_distance]
        acc_time = np.sqrt(2*acc_distances/acceleration)

        #the distances where we are decellerating
        decc_distances = distances[distances > half_distance] - half_distance

        #to get the decelleration times we will use the numpy solver 
        a = 0.5*decceleration
        b = max_velocity
        c = -1 * decc_distances

        decc_times = np.real(np.array([np.roots([a,b,ci])[1] for ci in c])) + half_time 
        decc_distances = decc_distances+half_distance

        # sort the decellerating times
        # indices = np.argsort(decc_times)
        # decc_times = decc_times[indices]
        # decc_distances = decc_distances[indices]

        # sort the accelerating times
        # indices = np.argsort(acc_time)
        # acc_time = acc_time[indices]
        # acc_distances = acc_distances[indices]

        # print('decc_distances', decc_distances)
        # print('decc_times', decc_times)

        # plots time vs distance 
        # plt.title(f'{field}')
        # plt.plot(acc_time, acc_distances, label='acceleration')
        # plt.plot(decc_times, decc_distances, label='decceleration')
        # plt.xlabel('time (s)')
        # plt.ylabel('distance (m)')
        # plt.legend()
        # plt.savefig(f'{field}_time_v_distance.pdf')
        # plt.show()

        total_time = np.concatenate([acc_time, decc_times])
        total_distances =  np.concatenate([acc_distances, decc_distances])

        acc_fields = self.distance_to_field(acc_distances)
        decc_fields = self.distance_to_field(decc_distances)
        
        total_fields = np.concatenate([acc_fields, decc_fields])

        # print(field)
        # print('total fields', total_fields)
        # print('total time', total_time)

        # plt.title(f'time v field {field}')
        # plt.scatter(acc_time, acc_fields, label='acceleration')
        # plt.scatter(decc_times, decc_fields, label='decceleration')
        # plt.plot(total_time, total_fields)
        # plt.xlabel('time (s)')
        # plt.ylabel('field (T)')
        # plt.legend()
        # plt.savefig(f'{field}_time_v_field.pdf')
        # plt.show()        

        # #time spent at each field going forwards
        field_centers = (total_fields[1:] + total_fields[:-1]) / 2
        time_taken = total_time[1:] - total_time[:-1]
        # print('sum time', np.sum(time_taken))


        # print('field', field)
        # print('Field centers ', field_centers)
        # print('total fields', total_fields)
        # print('time_taken', time_taken)
        # print('total time', sum(time_taken))
        
        # print('time_taken', time_taken)
        # print('time_taken sum ', sum(time_taken))

        #plot time at each field
        # plt.plot(time_taken, field_centers)
        # plt.scatter(time_taken, field_centers)
        # plt.xlabel('time (s)')
        # plt.ylabel('field (T)')
        # plt.savefig(f'{field}_DetlaTime_v_field.pdf')
        # plt.show()        

        return field_centers, time_taken

    def construct_all_trajectories(self, initial_velocity=0.):

        self.trajectory_time_at_fields = {}

        all_dists_master = self.fields_distances.T[0]
        all_fields_master = self.fields_distances.T[1]

        # there has been some confusion here 
        # I have used self.fields_distances instead of all_dists and all_fields
        # so I need to re-write this using these instead. 
        
        for field in self.sampled_fields: 
            # print(field)
            forwards = self.construct_single_trajectory(field)
            forwards_field_center, forwards_time_taken = forwards

            # since the path is symetrical we can just do the reverse of the forwards path
            backwards_field_center = np.flip(forwards_field_center)
            backwards_time_taken = np.flip(forwards_time_taken)

            self.trajectory_time_at_fields[field] = [forwards_field_center, forwards_time_taken, 
                                                    backwards_field_center, backwards_time_taken]

