## Intro

This is a controller node for turtlebot3. It allows robot to autonomously explore environment.

The algorithms are applied from [paper1](), [paper2]().

## How to play

1. install ros-noetic and turtlebot-related packages following [this e-mannual](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)
2. put src code in catkin_ws/src
3. catkin build

### For simulation

```
$ roslaunch turtlebot3_gazebo sim.launch
$ rosrun controller Controller3.py
```

### For real world

```
$ roscore [in PC]
$ roslaunch turtlebot3_bringup robot.launch [in turtlebot]
$ roslaunch turtlebot3_slam.launch [in PC]
$ roslaunch turtlebot3_navigation move_base.launch [in PC]
$ rosrun controller Controller3.py [in PC]
```

## Demos
