
import numpy as np
from tqdm import tqdm
import scipy 
import scipy.spatial
import matplotlib.pyplot as plt
import random

class AdaptiveSampler():
    def __init__(self, func, initial_points):
        
        self.objective = func 
        self.initial_points = initial_points
        self.points = initial_points
        self.evaluated_objective_funtion = []
        self.objective_args = []
        self.user_metric = None
        self.user_metric_args = []
        self.points_transform = None 
        self.points_transform_args = []
        
    def evalute_all_points(self):
        
        self.evaluated_objective_funtion = []
        for i,j in self.points:
            zi = self.objective(i,j, *self.objective_args)
            self.evaluated_objective_funtion.append(zi)
        
        self.evaluated_objective_funtion = np.array(self.evaluated_objective_funtion)
    
    def calculate_area_of_triangle(self, point1, point2, point3):
        # Convert the points to NumPy arrays for easier calculations
        p1 = np.array(point1)
        p2 = np.array(point2)
        p3 = np.array(point3)

        # Calculate two vectors in the plane of the triangle
        vector1 = p2 - p1
        vector2 = p3 - p1

        # Calculate the cross product of the two vectors
        cross_product = np.cross(vector1, vector2)

        # The magnitude of the cross product is the area of the triangle
        area = 0.5 * np.linalg.norm(cross_product)

        return area

    def less_first(self, a, b):
        return [a,b] if a < b else [b,a]

    def metric(self, x,y,z_x,z_y):
        dist = np.linalg.norm(x-y)
        abs_gradient = abs(z_x-z_y)
        return abs_gradient * dist 

    def metric_wrapper(self, x,y,z_x,z_y):
        # print('here,', self.user_metric)
        if self.user_metric == None:
            return self.metric(x,y,z_x,z_y)
        else:
            return self.user_metric(x,y,z_x,z_y, *self.user_metric_args)

    def get_delaunay(self,):
        # do the transform
        if self.points_transform == None:
            t_points = self.points  
        else:
            t_points = self.points_transform(self.points, *self.points_transform_args )

        tri = scipy.spatial.Delaunay(t_points)
        return tri, t_points

    def get_delaunay_edges(self):

        tri, t_points = self.get_delaunay()
        
        # get all the edges
        list_of_edges = []
        for triangle in tri.simplices:
            for e1, e2 in [[0,1],[1,2],[2,0]]: # for all edges of triangle
                list_of_edges.append(self.less_first(triangle[e1],triangle[e2])) 
        
        # take the unique ones
        array_of_edges = np.unique(list_of_edges, axis=0)
        return array_of_edges

    def get_triangle_z_angle(self, p1, p2, p3):

        vector1 = p2 - p1
        vector2 = p3 - p1
        z_axis_vector = np.array([0,0,1])

        # norm to triangle
        plane_norm = np.cross(vector1, vector2)
        plane_norm = plane_norm / np.linalg.norm(plane_norm)
        
        # angle 
        angle = np.arccos(np.dot(z_axis_vector, plane_norm))
        return angle

    def one_cycle(self):

        array_of_edges = self.get_delaunay_edges()
        
        # get the points with the highest score
        pairs = []
        list_of_metric = []
        for p1,p2 in array_of_edges:
            x = self.points[p1]
            y = self.points[p2]
            pairs.append([x,y])
            z_x = self.evaluated_objective_funtion[p1]
            z_y = self.evaluated_objective_funtion[p2]
            list_of_metric.append(self.metric_wrapper(x,y,z_x,z_y))
        
        # calculate the point as the half way point
        list_of_metric = np.array(list_of_metric)
        index = np.argmax(list_of_metric)
        new_point = (pairs[index][0] + pairs[index][1])/2
        new_evaluated_point = self.objective(new_point[0], new_point[1], *self.objective_args)
        
        return new_point, new_evaluated_point

    def one_cycle_area(self,):

        tri, t_points = self.get_delaunay()
        expanded_evalutated_points = np.expand_dims(self.evaluated_objective_funtion, axis=1)
        point_function = np.hstack([self.points, expanded_evalutated_points])

        point_sets = []
        areas = []

        for triangle in tri.simplices:
            current_points = [point_function[i] for i in triangle]            
            point_sets.append(current_points)
            current_area = self.calculate_area_of_triangle(*current_points)
            current_angle = self.get_triangle_z_angle(*current_points)
            metric = current_area*current_angle # switch*current_area + (1-switch)*current_area*current_angle
            areas.append(metric)

        # index for new point
        max_area_index = np.argmax(areas)
        # selected_points_index = random.choices(list(range(len(point_sets))), weights=areas)
        selected_points = point_sets[max_area_index]

        triangle_center = np.sum(selected_points, axis=0)/len(selected_points)
        new_point = np.array([triangle_center[0], triangle_center[1]])
        new_evaluated_point = self.objective(new_point[0], new_point[1], *self.objective_args)
        return new_point, new_evaluated_point
    

    def run_n_cycles(self, cycles):
        
        for i in tqdm(range(cycles)):
            new_point, new_evaluated_point = self.one_cycle_area()
            self.points = np.vstack([self.points, new_point])
            hstack = [self.evaluated_objective_funtion, new_evaluated_point]
            self.evaluated_objective_funtion = np.hstack(hstack)            
    
    def interpolate(self, resolution=20):
        
        points = self.points
        values = self.evaluated_objective_funtion
        x = points[:,0]
        y = points[:,1]
        
        xgrid = np.linspace(min(x), max(x),resolution)
        ygrid = np.linspace(min(y), max(y),resolution)
        xgrid, ygrid = np.meshgrid(xgrid, ygrid)
        interpolate = scipy.interpolate.griddata((x,y),
                        self.evaluated_objective_funtion,
                        (xgrid, ygrid),
                        method='cubic')
        
        self.interpolate_surface = interpolate
        self.interpolate_axis = [xgrid, ygrid]
    
    def plot(self,name):
        
        tri = scipy.spatial.Delaunay(self.points)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        fig.suptitle('Horizontally stacked subplots')
        x_axis = self.interpolate_axis[0]
        y_axis = self.interpolate_axis[1]
        X, Y = np.meshgrid(x_axis, y_axis)
        
        ax1.contourf(*self.interpolate_axis, self.interpolate_surface, levels=50, )
        ax2.contourf(*self.interpolate_axis, self.interpolate_surface, levels=50, )
        
        ax2.triplot(self.points[:,0], self.points[:,1], tri.simplices)
        ax2.plot(self.points[:,0], self.points[:,1], 'o')
        plt.savefig(name)
        plt.close()