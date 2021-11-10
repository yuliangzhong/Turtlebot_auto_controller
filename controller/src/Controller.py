#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, MapMetaData
import tf

import numpy as np
import os
import sys
import torch
import math
from scipy import ndimage

from Localmap_downsample import localmap_downsample

# model loading setup
sys.path.append(os.environ["HOME"] + "/git/plr-2021-learning-exploration")
path_to_model = os.environ["HOME"] + "/git/plr-2021-learning-exploration/policies/final.pt"
dtype = torch.float32
device = "cpu"


class Agent(object):
    def __init__(self):

        self.model_localmap_size = 50 # pixels.  local map size in pixels for trained model
        self.model_localmap_downsample_size = 10 # pixels.  downsampled map size in pixels for trained model
        self.goals_num = 10
        self.localmap_size = 50 # pixels, should <60, even

        self.map_sub = rospy.Subscriber('map', OccupancyGrid, self.map_read, queue_size=5)
        self.map_config = MapMetaData()
        self.map_data = []
        # self.map_2D = None

        self.costmap_sub = rospy.Subscriber('move_base/local_costmap/costmap', OccupancyGrid, self.costmap_read, queue_size=5)
        self.costmap_config = MapMetaData()
        self.costmap_data = []

        self.min_frontier_size = 8 # in pixels

        self.model = torch.load(path_to_model)
        print("model loaded")

    def map_read(self, msg):
        self.map_config = msg.info
        self.map_data = msg.data
    
    def global_map_2D(self):
        w = self.map_config.width
        h = self.map_config.height
        map_2D = np.zeros((h,w))
        index = w*(h-1)
        for i in range(h):
            map_2D[i] = self.map_data[index:index+w]
            index = index - w
        # self.map_2D = map_2D
        return map_2D
    
    def costmap_read(self, msg):
        self.costmap_config = msg.info
        self.costmap_data = msg.data
    
    # coordinates in ROS and python simulator are different!
    # transform to ROS coordinates, in meters
    def transform(self, data):
        result = np.zeros(data.shape)
        result[:,0] = data[:,1]/self.model_localmap_size*self.localmap_size*self.map_config.resolution
        result[:,1] = -data[:,0]/self.model_localmap_size*self.localmap_size*self.map_config.resolution
        result[:,2] = data[:,2] - math.pi/2
        return result

    # get localmap of robot global position (x,y)
    # and check if in local minimum
    def get_localmap(self, x, y): 
        # read the local map from top left cell
        # and assume the map is big enough to cut a local map, so no edge detection

        p = self.localmap_size
        cell_x_index = int((x - self.map_config.origin.position.x)/self.map_config.resolution - p/2)
        cell_y_index = int((y - self.map_config.origin.position.y)/self.map_config.resolution + p/2)
        localmap = np.ones((p, p))
        index = cell_y_index*self.map_config.width + cell_x_index # top left cell index
        for i in range(p):
            localmap[i]= self.map_data[index:index+p]
            index = index - self.map_config.width

        # stuck in mimimum
        is_in_local_minimum = False
        frontiers, _ = self.compute_frontiers(localmap, int(p/2), int(p/2))
        if(np.max(frontiers) == 0):
            is_in_local_minimum = True

        return localmap, is_in_local_minimum

    # if free, feasible. else, not feasible
    # global position (x,y)
    def check_move_feasible(self, x, y):
        x_index = int((x - self.map_config.origin.position.x)/self.map_config.resolution)
        y_index = int((y - self.map_config.origin.position.y)/self.map_config.resolution)
        index = y_index*self.map_config.width + x_index
        value = self.map_data[index]
        if(value==0):
            return True
        else:
            return False
    
    # given robot index position (x_ind,y_ind) in given 2D map (2d ndarray), compute frontiers
    def compute_frontiers(self, map, x_ind, y_ind):
        #print("start compute frontier")
        mask = (map == 0)
        labeled_image, _ = ndimage.measurements.label(mask)
        mask = labeled_image == labeled_image[x_ind,y_ind]  # get reachable area
        frontier_mask = ndimage.binary_dilation(mask) # expand reachable area for 1 block
        frontiers = (map == -1) & frontier_mask # get frontiers (True for frontiers, 0 for not)
        labeled_frontiers, num_frontiers = ndimage.measurements.label(frontiers,structure=[[1, 1, 1],
                                                                                           [1, 1, 1],
                                                                                           [1, 1, 1]])
        # remove too small frontiers, compute central position
        index_pos = []
        for i in range(num_frontiers):
            frontier_img = labeled_frontiers == i + 1
            if np.sum(frontier_img) < self.min_frontier_size:
                frontiers[frontier_img] = 0
            else:
                index = np.where(frontier_img==True)
                index_pos.append([index[0].mean(), index[1].mean()])
        #print(index_pos)
        #print("frontiers computed")
        return frontiers, np.array(index_pos)

    # generate a new goal in global ROS coordinate. Current robot position: (x,y)
    def new_goal(self, x, y):
        
        # to make sure we have heard data
        while(len(self.map_data)*len(self.costmap_data)==0):
            print("no data")
        
        # get maps and downsample
        localmap, is_in_local_minimum = self.get_localmap(x,y)
        localmap_zip, _ = localmap_downsample(localmap, ifonehot = True, zipsize = self.model_localmap_downsample_size)

        if(is_in_local_minimum):
            print("stucked!")
            # careful about coordinates transfer
            y_index = int((x - self.map_config.origin.position.x)/self.map_config.resolution)
            x_index = - int((y - self.map_config.origin.position.y)/self.map_config.resolution - self.map_config.height)
            print(x_index, y_index)
            _, ind_pos = self.compute_frontiers(self.global_map_2D(), x_index, y_index)

            if(len(ind_pos)==0):
                print("mission complete!")
                sys.exit()
                # exit this node

            ind_distance = np.ones(ind_pos.shape[0])
            for i in range(ind_pos.shape[0]):
                ind_distance[i] = (ind_pos[i][0]-x_index)**2 + (ind_pos[i][1]-y_index)**2
            print(ind_distance)
            pos = ind_pos*self.map_config.resolution
            closest_frontier = pos[np.argmin(ind_distance)]
            print(closest_frontier)
            global_x = self.map_config.origin.position.x + closest_frontier[1]
            global_y = self.map_config.height*self.map_config.resolution + self.map_config.origin.position.y - closest_frontier[0]
            print(global_x, global_y)
            best_goal = np.array([global_x - x, global_y - y, 0])
            print("global planner goal set!")
        else:
            # get goals from trained model
            z = torch.randn(self.goals_num, 3)  # hard coded latent dimension
            local_map_rep = np.repeat([localmap_zip], self.goals_num, axis=0)
            cond = torch.tensor(local_map_rep, dtype=dtype, device=device)
            self.model.eval().to(device=device)
            z_in = torch.cat((z, cond), 1)
            z_in.to(device=device)
            x_pred = self.model.decoder(z_in)
            x_pred = x_pred.detach().numpy().astype('float64')  # x_pred of dimension: goals_num x 3
            goals = self.transform(x_pred)
            
            # choose best goal via cost in costmap
            cost = np.zeros(self.goals_num)
            for i in range(self.goals_num):
                g = goals[i]
                posx_index = int(g[0]/self.costmap_config.resolution + self.costmap_config.width/2)
                posy_index = int(g[1]/self.costmap_config.resolution + self.costmap_config.height/2)
                cost_index = posy_index*self.costmap_config.width + posx_index
                cost[i] = self.costmap_data[cost_index]
                # don't choose the samples which are too closed to walls or not feasible
                if(cost[i]>50 or not self.check_move_feasible(g[0]+x,g[1]+y)): # cost map varies from 0 to 100
                    cost[i] = 0
            print(cost)
            if(sum(cost)==0):
                print("sum=0")
                best_goal = np.array([0,0,np.random.rand()*2*math.pi]) # resample
            else:
                best_goal = goals[np.argmax(cost)]
        
        # prepare to publish
        goalpub = PoseStamped()
        goalpub.header.frame_id = "map"
        goalpub.header.stamp = rospy.Time.now()
        goalpub.pose.position.x = best_goal[0] + x
        goalpub.pose.position.y = best_goal[1] + y
        goalpub.pose.orientation.z = math.sin(best_goal[2]/2)
        goalpub.pose.orientation.w = math.cos(best_goal[2]/2)

        print("publishing goal")
        return goalpub
        


if __name__ == "__main__":

    # set up
    rospy.init_node('turtlebot_controller', anonymous=True)
    listener = tf.TransformListener()
    new_goal = PoseStamped()
    agent = Agent()
    init_flag = True
    pub = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=10)


    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform('/map','/base_footprint',rospy.Time(0))
        except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        if(init_flag):
            #print("start publish")
            new_goal = agent.new_goal(trans[0],trans[1])
            
            rospy.sleep(1) 
            # https://answers.ros.org/question/306582/unable-to-publish-posestamped-message/
            pub.publish(new_goal)
            
            init_flag = False
            #print("exit")
        else:
            pos_error = (trans[0]-new_goal.pose.position.x)**2 + (trans[1]-new_goal.pose.position.y)**2
            if (pos_error < 0.0200):
                new_goal = agent.new_goal(trans[0],trans[1])
                rospy.sleep(1)
                pub.publish(new_goal)

    rospy.spin()
# chmod +x Controller.py

