#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, MapMetaData
from map_msgs.msg import OccupancyGridUpdate
from nav_msgs.srv import GetMap, GetMapRequest
import tf

import numpy as np
import os
import sys
import torch
import math
import time
from scipy import ndimage

from Localmap_downsample import localmap_downsample
from baseline_planner import LocalSamplingPlanner
from global_planner import GlobalPlanner
from robot import Robot
from config import Config

# model loading setup
sys.path.append(os.environ["HOME"] + "/git/plr-2021-learning-exploration")
path_to_model = os.environ["HOME"] + "/git/plr-2021-learning-exploration/policies/final.pt"
dtype = torch.float32
device = "cpu"

class Agent(object):
    def __init__(self):

        self.planner = "policy_planner" # "uniform_planner" or "policy_planner"

        # map service
        self.map_srv = rospy.ServiceProxy('/dynamic_map', GetMap) # Create the connection to the service
        self.map_config = MapMetaData()
        self.map_data = None

        # Caution! costmap only update once!
        # https://answers.ros.org/question/289710/global-costmap-updating-mysteriously/
        self.costmap_sub = rospy.Subscriber('move_base/global_costmap/costmap', OccupancyGrid, self.costmap_read, queue_size=10)
        self.costmap_update_sub = rospy.Subscriber('move_base/global_costmap/costmap_updates', OccupancyGridUpdate, self.costmap_update, queue_size=10)
        self.costmap_config = MapMetaData()
        self.costmap_data = None
        
        # config for uniform planner
        self.cfg = Config()
        self.cfg.voxel_size = 0.05 # m/voxel
        self.cfg.local_submap_size_x = 50
        self.cfg.local_submap_size_y = 50
        self.cfg.velocity_max = 0.26
        self.cfg.acceleration = 10
        self.cfg.yaw_acceleration = 10 # big enough
        self.cfg.yaw_rate_max = 1.82
        self.cfg.camera_range = 3 # m
        self.cfg.camera_fov = 360
        self.cfg.glob_radius = 0.3
        self.cfg.glob_min_frontier_size = 8

        self.robot = Robot(self.cfg)

        # config for uniform planner
        self.sampling_steps = 20
        self.sampling_yaw = 1
        self.action_bounds = [self.cfg.local_submap_size_x/2, self.cfg.local_submap_size_y/2, math.pi]
        self.count_only_feasible_samples = False
        self.use_normalized_gain = False
        self.uniform_planner = LocalSamplingPlanner(self.robot, self.sampling_steps, self.sampling_yaw, 
                                        self.action_bounds, self.count_only_feasible_samples, self.use_normalized_gain)

        
        self.global_planner = GlobalPlanner(self.cfg, self.robot)

        # config for policy planner
        self.model_localmap_size = 50 # pixels.  local map size in pixels for trained model
        self.model_localmap_downsample_size = 10 # pixels.  downsampled map size in pixels for trained model
        self.min_frontier_size = 8

        self.goals_num = 20
        self.criteria = "voxels" # "distance", "voxels", "gains"
        if(self.planner == "policy_planner"): # save time
            self.model = torch.load(path_to_model)
            print("model loaded")
        
        # time and steps
        self.use_ros_time = True # True or False
        self.explore_steps = 0
        self.plan_start_time = 0
        self.plan_time_buffer = 0
        self.plan_time = []
        self.map_load = []
        self.check_minimum = []
        self.global_plan_time = []
        self.local_plan_time = []
        self.safe_goal_time = []
        
    
    def get_current_time(self):
        if(self.use_ros_time):
            return rospy.Time.now().to_sec()
        return time.time()

    def global_map_2D(self):
        # in ROS, [free, wall, unobserved] = [0,100,-1]
        # in 2d sim, [free, wall, unobserved] = [0,1,2]
        self.map_data = np.array(self.map_data)
        self.map_data[self.map_data==100]=1
        self.map_data[self.map_data==-1]=2

        w = self.map_config.width
        h = self.map_config.height
        map_2D = np.zeros((h,w))
        index = w*(h-1)
        for i in range(h):
            map_2D[i] = self.map_data[index:index+w]
            index = index - w
        return map_2D

    def costmap_read(self, msg):
        self.costmap_config = msg.info
        self.costmap_data = np.array(msg.data)
    
    def costmap_update(self, msg):
        Width = self.costmap_config.width
        for i in range(msg.height):
            self.costmap_data[(msg.y+i)*Width+msg.x: (msg.y+i)*Width+msg.x+msg.width] = msg.data[i*msg.width:(i+1)*msg.width]
    
    def cost_map_2D(self):
        w = self.costmap_config.width
        h = self.costmap_config.height
        map_2D = np.zeros((h,w))
        index = w*(h-1)
        for i in range(h):
            map_2D[i] = self.costmap_data[index:index+w]
            index = index - w
        return map_2D
    
    # trans (x,y) in ros coordinates --> numpy index coordinates
    def ROS_2_MapInd(self, x, y): # (x,y) global co in meters
        x_ind = int((-y + self.map_config.height*self.map_config.resolution + self.map_config.origin.position.y)/self.map_config.resolution)
        y_ind = int((x - self.map_config.origin.position.x)/self.map_config.resolution)
        return x_ind, y_ind
    
    def MapInd_2_ROS(self, x_ind, y_ind):
        x = y_ind*self.map_config.resolution + self.map_config.origin.position.x
        y = self.map_config.height*self.map_config.resolution + self.map_config.origin.position.y - x_ind*self.map_config.resolution
        return x,y
    
    # cost map should have same size of global map
    def safe_goal(self, best_goal, r=3):
        x_ind, y_ind = self.ROS_2_MapInd(best_goal[0], best_goal[1])
        costmap = self.cost_map_2D()
        cost = costmap[x_ind,y_ind]
        if(cost<60):
            return best_goal
        else:
            print("[INFO] Dangerous goal, safe once!")
            costs = costmap[x_ind-r:x_ind+r, y_ind-r:y_ind+r]
            argi = np.argmin(costs)
            xx = int(argi/(2*r))
            yy = argi-xx*(2*r)
            best_goal[0],best_goal[1] = self.MapInd_2_ROS(x_ind-r+xx, y_ind-r+yy)

            if(costs[xx,yy]>65):
                print("[INFO] Safe again!")        
                best_goal = self.safe_goal(best_goal)
            return best_goal
    
    # given robot index position (x_ind,y_ind) in given 2D map (2d ndarray), compute frontiers
    def compute_frontiers_pos(self, map, x_ind, y_ind):
        mask = (map == 0)
        labeled_image, _ = ndimage.measurements.label(mask)
        mask = labeled_image == labeled_image[x_ind,y_ind]  # get reachable area
        frontier_mask = ndimage.binary_dilation(mask) # expand reachable area for 1 block
        frontiers = (map == 2) & frontier_mask # get frontiers (True for frontiers, 0 for not)
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
        return np.array(index_pos)

    def policy_plan(self):

        localmap = self.robot.get_local_submap()
        localmap_zip, _ = localmap_downsample(localmap, ifonehot= True, zipsize= self.model_localmap_downsample_size)

        # get goals from trained model
        z = torch.randn(self.goals_num, 3)  # hard coded latent dimension
        local_map_rep = np.repeat([localmap_zip], self.goals_num, axis=0)
        cond = torch.tensor(local_map_rep, dtype=dtype, device=device)
        self.model.eval().to(device=device)
        z_in = torch.cat((z, cond), 1).to(device=device)
        x_pred = self.model.decoder(z_in)
        x_pred = x_pred.detach().numpy().astype('float64')  # x_pred of dimension: goals_num x 3
        
        # Caution! zoom localmap size!
        samples = np.zeros(x_pred.shape)
        samples[:,0] = x_pred[:,0]/self.model_localmap_size*self.cfg.local_submap_size_x
        samples[:,1] = x_pred[:,1]/self.model_localmap_size*self.cfg.local_submap_size_y
        samples[:,2] = x_pred[:,2]

        # check feasibility and compute gains
        voxels = np.zeros(self.goals_num)
        gains = np.zeros(self.goals_num)
        frontier_distance = 1e10*np.ones(self.goals_num)
        ind_pos = self.compute_frontiers_pos(self.robot.observed_map, self.robot.position_x, self.robot.position_y)
        valid_goal_flag = False
        for i in range(self.goals_num):
            [xi, yi, yawi] = samples[i]
            _, safe_flag = self.robot.check_move_feasible(xi,yi,0.25)
            if(safe_flag):
                valid_goal_flag = True
                if(self.criteria == "voxels" or self.criteria == "gains"):
                    voxels[i] = self.uniform_planner.get_number_of_visible_voxels(xi,yi,yawi)
                    gains[i] = self.uniform_planner.compute_gain(xi,yi,yawi,voxels[i])
                else:
                    for pos in ind_pos:
                        frontier_distance[i] = min(frontier_distance[i],((pos[0]-self.robot.position_x - xi)**2 + (pos[1]-self.robot.position_y - yi)**2))
            # else voxels = gains = 0, distance = 1e10
        
        if(self.criteria == "voxels"):
            result = samples[np.argmax(voxels)]
        elif(self.criteria == "gains"):
            result = samples[np.argmax(gains)]
        else:
            result = samples[np.argmin(frontier_distance)]
        
        # Make sure get at least one goal
        if(not valid_goal_flag):
            print("WARNING: no goal")
            result = self.policy_plan()

        return result
    

    # generate a new goal in global ROS coordinate. Current robot position: (x,y)
    def new_goal(self, x, y):

        self.explore_steps = self.explore_steps + 1

        # time stamp
        self.plan_time_buffer = self.get_current_time()
        self.plan_start_time = self.plan_time_buffer

        end_flag = False

        # load updated map
        result = self.map_srv(GetMapRequest()) # Call the service
        self.map_config = result.map.info
        self.map_data = result.map.data

        # time stamp
        tmap = self.get_current_time()-self.plan_time_buffer
        self.map_load.append(tmap)
        print(f'[INFO] load map for {tmap:.4f} seconds')
        self.plan_time_buffer = self.get_current_time()

        # to make sure we have heard data
        while(len(self.map_data)*len(self.costmap_data)==0):
            print("WARNING: No data")

        # get data
        self.robot.observed_map = self.global_map_2D()
        self.robot.position_x, self.robot.position_y = self.ROS_2_MapInd(x,y)
        is_in_local_minimum = self.global_planner.is_local_minimum(self.robot.get_local_submap(), 
                                                                  [self.cfg.local_submap_size_x / 2, self.cfg.local_submap_size_y / 2])
        # time stamp
        tcheck = self.get_current_time()-self.plan_time_buffer
        self.check_minimum.append(tcheck)
        print(f'[INFO] check minimum for {tcheck:.4f} seconds')
        self.plan_time_buffer = self.get_current_time()

        if(is_in_local_minimum):
            print("[INFO] Robot is in local minimum")

            result = self.global_planner.plan(True)

            # time stamp
            tglobal = self.get_current_time()-self.plan_time_buffer
            self.global_plan_time.append(tglobal)
            print(f'[INFO] get global goal for {tglobal:.4f} seconds')
            self.plan_time_buffer = self.get_current_time()

            if(result.success):
                goalx, goaly = self.MapInd_2_ROS(result.x, result.y)
                best_goal = np.array([goalx, goaly, result.yaw - math.pi/2])
            else:
                end_flag = True

        else:

            if(self.planner == "uniform_planner"):
                # already ensured in .plan() for at least one result
                plan,_,_ = self.uniform_planner.plan()

                # time stamp
                tlocal = self.get_current_time()-self.plan_time_buffer
                self.local_plan_time.append(tlocal)
                print(f'[INFO] get local goal for {tlocal:.4f} seconds')
                self.plan_time_buffer = self.get_current_time()

                # Caution! coordinate change here
                goalx = plan.y*self.map_config.resolution + x
                goaly = -plan.x*self.map_config.resolution + y
                yaw = plan.yaw - math.pi/2
                best_goal = np.array([goalx, goaly, yaw])

            elif(self.planner == "policy_planner"):
                # already ensured in .plan() for at least one result
                goal = self.policy_plan()

                # time stamp
                tlocal = self.get_current_time()-self.plan_time_buffer
                self.local_plan_time.append(tlocal)
                print(f'[INFO] get local goal for {tlocal:.4f} seconds')
                self.plan_time_buffer = self.get_current_time()

                # Caution! change coordinates!
                goalx = x + goal[1]*self.map_config.resolution
                goaly = y - goal[0]*self.map_config.resolution
                yaw = goal[2] - math.pi/2
                best_goal = np.array([goalx, goaly, yaw])

            else:
                print("ERROR: Planner name error! Exiting...")
                sys.exit()
        
        # prepare to publish
        goalpub = PoseStamped()
        # check safety
        if(not end_flag):
            best_goal = self.safe_goal(best_goal)
            
            # time stamp
            tsafe = self.get_current_time()-self.plan_time_buffer
            self.safe_goal_time.append(tsafe)
            print(f'[INFO] safe goal for {tsafe:.4f} seconds')
            self.plan_time_buffer = self.get_current_time()

            goalpub.header.frame_id = "map"
            goalpub.header.stamp = rospy.Time.now()
            goalpub.pose.position.x = best_goal[0]
            goalpub.pose.position.y = best_goal[1]
            goalpub.pose.orientation.z = math.sin(best_goal[2]/2)
            goalpub.pose.orientation.w = math.cos(best_goal[2]/2)

            # time stamp
            t = self.get_current_time()-self.plan_start_time
            print(f'[INFO] Generate new goal for total {t:.4f} seconds')
            self.plan_time.append(t)

        return goalpub, end_flag



if __name__ == "__main__":

    rospy.init_node('turtlebot_controller', anonymous=True)
    print("Controller starting...")

    listener = tf.TransformListener()
    new_goal = PoseStamped()
    agent = Agent()
    init_flag = True
    pub = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=10)
    time_buffer = 0
    start_time = 0
    print("Planner: ", agent.planner)

    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform('/map','/base_footprint',rospy.Time(0))
        except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        
        result = agent.map_srv(GetMapRequest()) # Call the service
        agent.map_config = result.map.info
        agent.map_data = result.map.data

        if(init_flag):
            print("Start exploration!")

            time_buffer = agent.get_current_time()
            start_time = time_buffer
            
            new_goal, end_flag = agent.new_goal(trans[0],trans[1])
            
            if(end_flag):
                print("Already explored! Existing...")
                sys.exit()

            while(pub.get_num_connections()==0):
                print("WARNING: No listener")
            pub.publish(new_goal)

            init_flag = False
            
        else:
            
            pos_error = (trans[0]-new_goal.pose.position.x)**2 + (trans[1]-new_goal.pose.position.y)**2
            gap_time = agent.get_current_time() - time_buffer

            if(gap_time > 120):
                print("WARNING: Robot stucked or infeasible goal!")
                pos_error = 0
                end_flag = True

            if (pos_error < 0.0500):

                print("[INFO] Goal reached!")
                
                time_buffer = agent.get_current_time()
                new_goal, end_flag = agent.new_goal(trans[0],trans[1])

                if(end_flag):
                    total_time = agent.get_current_time() - start_time
                    av_plan_time = np.average(agent.plan_time)
                    av_map_load = np.average(agent.map_load)
                    av_check_min = np.average(agent.check_minimum)
                    av_global_time = np.average(agent.global_plan_time)
                    av_local_time = np.average(agent.local_plan_time)
                    av_safe_time = np.average(agent.safe_goal_time)
                    print("##################################")
                    print(f'Total execution time: {total_time:.4f}')
                    print(f'Total new goal steps: {agent.explore_steps}')
                    print(f'Average planning time: {av_plan_time:.4f}')
                    print(f'Average map load time: {av_map_load:.4f}')
                    print(f'Average check time: {av_check_min:.4f}')
                    print(f'Average global time: {av_global_time:.4f}')
                    print(f'Average local time: {av_local_time:.4f}')
                    print(f'Average safe time: {av_safe_time:.4f}')
                    print("Mission complete! Exiting...")

                    sys.exit()

                while(pub.get_num_connections()==0):
                    print("WARNING: No listener")
                pub.publish(new_goal)

    rospy.spin()

# chmod +x Controller.py
# publisher buffer
# https://answers.ros.org/question/306582/unable-to-publish-posestamped-message/
# tuning move_base:
# https://kaiyuzheng.me/documents/navguide.pdf


# Warning: TF_REPEATED_DATA ignoring data with redundant timestamp for frame base_footprint at time 321.026000 according to authority /gazebo
# DWA local planner sometimes doesn't work well

