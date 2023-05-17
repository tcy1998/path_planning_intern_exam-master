import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Node class to represent a node in the RRT tree
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

# RRT class to implement the RRT algorithm
class RRT:
    def __init__(self, start, goal, circle_obstacle_list, poly_obstacle_list, rand_area, expand_dis=0.5, goal_sample_rate=10, max_iter=500):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.min_rand_x = rand_area[0]
        self.max_rand_x = rand_area[1]
        self.min_rand_y = rand_area[2]
        self.max_rand_y = rand_area[3]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.circle_obstacle_list = circle_obstacle_list
        self.poly_obstacle_list = poly_obstacle_list
        self.node_list = [self.start]

    # Main method to plan a path from the start to the goal
    def planning(self):
        for i in range(self.max_iter):
            if random.randint(0, 100) > self.goal_sample_rate:
                rnd = self.get_random_node()
            else:
                rnd = self.goal

            nearest_ind = self.get_nearest_node_index(rnd)                  # Find nearest node
            nearest_node = self.node_list[nearest_ind]                      

            new_node = self.steer(nearest_node, rnd, self.expand_dis)       # Steer the node to a closet range 

            if self.check_collision_circle(new_node, self.circle_obstacle_list):          # Check collision for circle
                if self.check_collision_poly(new_node, self.poly_obstacle_list):
                    self.node_list.append(new_node)

            if self.calc_distance_and_angle(new_node, self.goal)[0] <= self.expand_dis:
                final_node = self.steer(new_node, self.goal, self.expand_dis)
                if self.check_collision_circle(final_node, self.circle_obstacle_list):
                    if self.check_collision_poly(final_node, self.poly_obstacle_list):
                        return self.generate_final_course(len(self.node_list) - 1)

            # if i % 5 == 0:
            #     self.draw_graph(rnd)

        return None

    # Connect two nodes with a line segment
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)
        new_node.parent = from_node

        return new_node

    # Generate a random node
    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(random.uniform(self.min_rand_x, self.max_rand_x),
                       random.uniform(self.min_rand_y, self.max_rand_y))
        else:
            rnd = Node(self.goal.x, self.goal.y)

        return rnd

    # Find the index of the node in the tree that is closest to a given node
    def get_nearest_node_index(self, node):
        dlist = [(n.x - node.x) ** 2 + (n.y - node.y) ** 2 for n in self.node_list]
        min_index = dlist.index(min(dlist))

        return min_index

    # Calculate the Euclidean distance and angle between two nodes
    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.sqrt(dx ** 2 + dy ** 2)
        theta = math.atan2(dy, dx)
        return d, theta

    # Check if a node collides with any obstacle in the obstacle list
    def check_collision_circle(self, node, circle_obstacle_list):
        for (ox, oy, size) in circle_obstacle_list:
            dx = ox - node.x
            dy = oy - node.y
            d = math.sqrt(dx ** 2 + dy ** 2)
            if d <= size:
                return False

        return True
    
    def check_collision_poly(self, node, polygon_obstacle_list):
        for i, obstacle in enumerate(polygon_obstacle_list):
            x, y = node.x, node.y
            n = len(obstacle)
            # inside = False
            p1x, p1y = obstacle[0]
            for i in range(n + 1):
                p2x, p2y = obstacle[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= x_inters:
                                # inside = not inside
                                return False
                p1x, p1y = p2x, p2y
            return True

    # Generate the final path from the start to the goal
    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

    # Draw the RRT tree
    def draw_graph(self, rnd=None):
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")

        for node in self.node_list:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")

        for (ox, oy, size) in self.circle_obstacle_list:
            self.plot_circle(ox, oy, size)

        for i, obstacle in enumerate(self.poly_obstacle_list):
            self.plot_polygon(obstacle, 'darkgray')

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        plt.axis([self.min_rand_x, self.max_rand_x, self.min_rand_y, self.max_rand_y])
        plt.grid(True)
        plt.pause(0.01)

    # Plot a circle obstacle
    def plot_circle(self, x, y, size):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(math.radians(d)) for d in deg]
        yl = [y + size * math.sin(math.radians(d)) for d in deg]
        plt.plot(xl, yl)

    # Define functions to plot obstacles
    def plot_polygon(self, polygon, color):
        x, y = zip(*polygon)
        plt.fill(x, y, color=color)
    
def main():
    # ====Search Path with RRT====
    start = (-2, -2)
    goal = (8, 6)
    circle_obstacle_list = [(2, 2, 1), (3, 3, 1), (4, 4, 1)]

    # Define the area
    rand_area = [-5, 10, -5, 10]
    static_obstacles = [
                        [(2, 2), (2, 8), (3, 8), (3, 3), (8, 3), (8, 2)],
                        [(6, 6), (7, 6), (7, 7), (6, 7)]
                        ]

    # Plan the path using the RRT algorithm
    rrt = RRT(start, goal, circle_obstacle_list, static_obstacles, rand_area)
    path = rrt.planning()

    # Plot the final path and the RRT tree
    if path is None:
        print("Cannot find path")
    else:
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        rrt.draw_graph()
        plt.show()

if __name__ == '__main__':
    main()
