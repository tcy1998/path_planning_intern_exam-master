import matplotlib.pyplot as plt
import math
import random
from matplotlib.animation import FuncAnimation
from rrt import RRT
from rrt_star import RRTStar

# obstacle type flag
static_only = 0


# Calculate the distance between two points
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Generate the neighbors of a point
def neighbors(point):
    x, y = point
    return [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
            (x, y - 1), (x, y + 1),
            (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)]


# Check if a point is inside a polygon
def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


# Generate a path from start to goal avoiding static and dynamic obstacles
def generate_path():
    return

# Define the start and goal points
start = (-2, -2)
goal = (8, 6)

# Define the static obstacles as a list of polygons
static_obstacles = [
    [(2, 2), (2, 8), (3, 8), (3, 3), (8, 3), (8, 2)],
    [(6, 6), (7, 6), (7, 7), (6, 7)]
]


# Define the dynamic obstacles as a list of points
dynamic_obstacles = [
    {'initial_position': [
        (10, 1)], "velocity": [random.uniform(-1, 1), random.uniform(-1, 1)]},
    {'initial_position': [
        (2.5, 10)], "velocity": [random.uniform(-1, 1)*.5, random.uniform(-1, 1)*.5]},
    {'initial_position': [
        (5, 5)], "velocity": [random.uniform(-1, 1)*.2, random.uniform(-1, 1)*.2]},
    {'initial_position': [
        (0, 2.5)], "velocity": [random.uniform(-1, 1)*.1, random.uniform(-1, 1)*.1]}
]

# Define functions to plot obstacles
def plot_polygon(polygon, color):
    x, y = zip(*polygon)
    axes.fill(x, y, color=color)


def get_dynamic_obstacle_location(obstacle, frame):
    point = obstacle['initial_position']
    velocity = obstacle['velocity']
    vx, vy = velocity[0], velocity[1]
    x = [i[0] + frame*vx for i in point]
    y = [i[1] + frame*vy for i in point]
    return x, y


fig = plt.figure(figsize=(5, 5))
axes = fig.add_subplot(111)
plt.xlim(-5, 15)
plt.ylim(-5, 15)
plt.xlabel('X')
plt.ylabel('Y')

dynamic_obstacles_location = []

for i, obstacle in enumerate(dynamic_obstacles):
    point, = axes.plot([], [], 'ok', ms=20)
    dynamic_obstacles_location.append(point)


for i, obstacle in enumerate(static_obstacles):
    plot_polygon(obstacle, 'darkgray')


def update_animation(frame):

    # update dynamic obstacles
    circle_obstacle = []
    r = 0.5
    global new_start
    for i, obstacle in enumerate(dynamic_obstacles):
        x, y = get_dynamic_obstacle_location(obstacle, frame+1)
        dynamic_obstacles_location[i].set_data(x, y)
        circle_obstacle.append((x[0],y[0],r))

    solver = 'RRT'
    # solver = 'RRT*' 
    # TODO: you may compute the path here!
    if solver == 'RRT':
        rrt = RRT(new_start, goal, circle_obstacle, static_obstacles, rand_area = [-5, 15, -5, 15], expand_dis=0.35, goal_sample_rate=10, max_iter=300)
        path = rrt.planning()                       # Get the path from RRT
        if path is None:
            print('No path found, Hold position')
            path = [new_start]
            for ii in range(frame):
                path.append(new_start)              # If no new path is found, hold position
        else:
            path = path[::-1]                       # Reverse the path from RRT
        if len(path) <= frame:                         
            for ii in range(frame-len(path)+1):     # If the path is shorter than the current frame, add the last point to avoid index error
                path.append(path[-1])              
        print(len(path), frame)
        new_start = (path[frame])                   # Start from the current position

    elif solver == 'RRT*':
        rrt_star = RRTStar(new_start, goal, circle_obstacle, static_obstacles, rand_area = [-5, 15, -5, 15], expand_dist=0.35, goal_sample_rate=10, max_iter=300, max_distance=1.0)
        path = rrt_star.plan()                      # Get the path from RRT*
        if path is None:
            print('No path found, Hold position')
            path = [new_start]
            for ii in range(frame):
                path.append(new_start)              # If no new path is found, hold position
        else:
            path = path[::-1]                       # Reverse the path from RRT_star
        if len(path) <= frame:                         
            for ii in range(frame-len(path)+1):     # If the path is shorter than the current frame, add the last point to avoid index error
                path.append(path[-1])              
        print(len(path), frame)
        new_start = (path[frame])                   # Start from the current position


    # Plot the path as a red line up to the current frame
    x = [i[0] for i in path[:frame+1]]
    y = [i[1] for i in path[:frame+1]]
    axes.plot(x, y, color='red')

    # Plot the start and goal points as green and blue circles, respectively
    axes.scatter(start[0], start[1], color='green', s=100)
    axes.scatter(goal[0], goal[1], color='blue', s=100)
    return []


# Create the animation using FuncAnimation
global new_start
new_start = start
animation = FuncAnimation(fig, update_animation, frames=45, interval=250, blit=True)

# Show the plot
plt.show()
