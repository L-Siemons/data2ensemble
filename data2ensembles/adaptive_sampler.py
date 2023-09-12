
import numpy as np
from tqdm import tqdm
import scipy 
import scipy.spatial
import matplotlib.pyplot as plt

class AdaptiveSampler():
    def __init__(self, func, initial_points):
        
        self.objective = func 
        self.initial_points = initial_points
        self.points = initial_points
        self.evaluated_objective_funtion = []
        self.objective_args = []
        self.user_metric = None
        self.user_metric_args = []
        
    def evalute_all_points(self):
        
        self.evaluated_objective_funtion = []
        for i,j in self.points:
            zi = self.objective(i,j, *self.objective_args)
            self.evaluated_objective_funtion.append(zi)
        
        self.evaluated_objective_funtion = np.array(self.evaluated_objective_funtion)
    
    def less_first(self, a, b):
        return [a,b] if a < b else [b,a]

    def metric(self, x,y,z_x,z_y):
        dist = np.linalg.norm(x-y)
        abs_gradient = abs(z_x-z_y)
        return abs_gradient * dist 

    def metric_wrapper(self, x,y,z_x,z_y):

        if self.user_metric == None:
            return self.metric(x,y,z_x,z_y)
        else:
            return self.user_metric(x,y,z_x,z_y, *self.user_metric_args)
    
    def one_cycle(self):
        
        # do the transform
        tri = scipy.spatial.Delaunay(self.points)
        
        # get all the edges
        list_of_edges = []
        for triangle in tri.simplices:
            for e1, e2 in [[0,1],[1,2],[2,0]]: # for all edges of triangle
                list_of_edges.append(self.less_first(triangle[e1],triangle[e2])) 
        
        # take the unique ones
        array_of_edges = np.unique(list_of_edges, axis=0)
        
        # get the points with the highest score
        pairs = []
        list_of_metric = []
        for p1,p2 in array_of_edges:
            x = tri.points[p1]
            y = tri.points[p2]
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
    
    def run_n_cycles(self, cycles):
        
        for i in range(cycles):
            new_point, new_evaluated_point = self.one_cycle()
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
                        method='linear')
        
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