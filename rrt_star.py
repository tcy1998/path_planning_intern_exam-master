import random
import math
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0

class RRTStar:
    def __init__(self, start, goal, circle_obstacle_list, poly_obstacle_list, rand_area, max_iter=1000, expand_dist=0.2, goal_sample_rate=5, max_distance=1.0):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.min_rand_x = rand_area[0]
        self.max_rand_x = rand_area[1]
        self.min_rand_y = rand_area[2]
        self.max_rand_y = rand_area[3]
        
        self.circle_obstacle_list = circle_obstacle_list
        self.poly_obstacle_list = poly_obstacle_list
        self.obstacle_list = circle_obstacle_list
        self.max_iter = max_iter
        self.expand_dist = expand_dist
        self.goal_sample_rate = goal_sample_rate
        self.max_distance = max_distance

    def plan(self):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd = self.get_random_node()                                # Get a random node
            nearest_index = self.get_nearest_node_index(rnd)            # Find the nearest node index
            nearest_node = self.node_list[nearest_index]                # Get the nearest node

            new_node = self.steer(nearest_node, rnd)                    # Steer from the random node based on the nearest node

            if self.check_collision(new_node, self.circle_obstacle_list, self.poly_obstacle_list):          # Check if the new node collides with an obstacle
                near_indices = self.find_near_nodes(new_node)                                               # Find the near nodes in a radius of max_distance
                node_with_min_cost = self.choose_parent(new_node, near_indices)                             # Choose the parent node with the lowest cost
                self.node_list.append(new_node)                                                             # Add the new node to the node list  
                self.rewire(node_with_min_cost, new_node, near_indices)                                     # Rewire the tree if a better path is found      
            
            # if i % 20 == 0:
            #     self.draw_graph(rnd)                                                                        # Draw the graph every 5 iterations
        
        last_index = self.get_best_last_index()                                                             # Get the index of the node with the lowest cost                 
        if last_index is None:
            return None

        path = self.generate_final_course(last_index)                                                       # Generate the final path
        return path
    
    # Returns a random node within the specified bounds.
    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(random.uniform(self.min_rand_x, self.max_rand_x),
                       random.uniform(self.min_rand_y, self.max_rand_y))
        else:
            rnd = Node(self.goal.x, self.goal.y)       

        return rnd

    # Connect two nodes with a line segment and calculate the cost
    def steer(self, from_node, to_node):
        new_node = Node(from_node.x, from_node.y)
        d = math.sqrt((to_node.x - new_node.x) ** 2 + (to_node.y - new_node.y) ** 2)
        theta = math.atan2(to_node.y - new_node.y, to_node.x - new_node.x)
        new_node.x += self.expand_dist * math.cos(theta)
        new_node.y += self.expand_dist * math.sin(theta)
        new_node.parent = from_node
        new_node.cost = from_node.cost + d
        return new_node

    # Chooses the parent node that results in the minimum cost for a given node.
    def choose_parent(self, new_node, near_indices):
        if len(near_indices) == 0:
            return None

        costs = []
        for i in near_indices:
            near_node = self.node_list[i]
            if self.check_collision(new_node, self.obstacle_list, self.poly_obstacle_list):
                d = math.sqrt((near_node.x - new_node.x) ** 2 + (near_node.y - new_node.y) ** 2)
                cost = near_node.cost + d
                costs.append(cost)
            else:
                costs.append(float("inf"))

        min_cost = min(costs)
        if min_cost == float("inf"):
            return None

        min_index = near_indices[costs.index(min_cost)]
        new_node.cost = min_cost
        new_node.parent = self.node_list[min_index]
        return self.node_list[min_index]

    # Rewires the tree by updating the parent and cost of nearby nodes if a better path is found.
    def rewire(self, node_with_min_cost, new_node, near_indices):
        for i in near_indices:
            near_node = self.node_list[i]
            d = math.sqrt((near_node.x - new_node.x) ** 2 + (near_node.y - new_node.y) ** 2)
            scost = new_node.cost + d
            if near_node.cost > scost and self.check_collision(new_node, self.circle_obstacle_list, self.poly_obstacle_list):           
                near_node.parent = new_node
                near_node.cost = scost
                self.propagate_cost_to_leaves(near_node)

    # Propagates the cost of a node to all of its descendants. 
    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                d = self.node_distance(node, parent_node)
                node.cost = parent_node.cost + d
                self.propagate_cost_to_leaves(node)

    # Returns the indices of nodes that are within a certain distance of a given node.
    def find_near_nodes(self, new_node):
        n = len(self.node_list) + 1
        r = self.max_distance * math.sqrt((math.log(n) / n))
        indices = []
        for i, node in enumerate(self.node_list):
            if self.node_distance(node, new_node) <= r:
                indices.append(i)
        return indices

    # Find the index of the node in the tree that is closest to a given node
    def get_nearest_node_index(self, node):
        dlist = [(n.x - node.x) ** 2 + (n.y - node.y) ** 2 for n in self.node_list]
        min_index = dlist.index(min(dlist))                 

        return min_index

    # Check if a node collides with any obstacle in the obstacle list
    def check_collision(self, node, circle_obstacle_list, poly_obstacle_list):
        for i, obstacle in enumerate(poly_obstacle_list):
            x, y = node.x, node.y
            n = len(obstacle)
            p1x, p1y = obstacle[0]
            if_Safe_poly = True
            for i in range(n + 1):
                p2x, p2y = obstacle[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= x_inters:
                                if_Safe_poly = not if_Safe_poly                  
                p1x, p1y = p2x, p2y
            if if_Safe_poly == False:
                break
        
        for (ox, oy, size) in circle_obstacle_list:
            IF_Safe_circle = True
            dx = ox - node.x
            dy = oy - node.y
            d = math.sqrt(dx ** 2 + dy ** 2)
            if d <= size:
                IF_Safe_circle = False  # collision
                break

            
        IF_Safe = IF_Safe_circle and if_Safe_poly
        return IF_Safe 

    def get_best_last_index(self):
        goal_node = Node(self.goal.x, self.goal.y)
        n = len(self.node_list)
        r = self.max_distance * math.sqrt((math.log(n) / n))
        near_goal_indices = []
        for i, node in enumerate(self.node_list):
            if self.node_distance(node, goal_node) <= r:
                near_goal_indices.append(i)

        if len(near_goal_indices) == 0:
            return None

        min_cost = min([self.node_list[i].cost for i in near_goal_indices])
        for i in near_goal_indices:
            if self.node_list[i].cost == min_cost:
                return i
        return None

    # Generate the final path from the start to the goal
    def generate_final_course(self, goal_index):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_index]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path

    # Distance between points and nodes
    def distance(self, n1, n2):
        return math.sqrt((n1.x - n2[0]) ** 2 + (n1.y - n2[1]) ** 2)
    
    # Distance between two nodes
    def node_distance(self, n1, n2):
        return math.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2)
    
    # Draw the RRT tree
    def draw_graph(self, rnd=None):
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")

        for node in self.node_list:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g", linewidth=0.2)

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
    start = (-2, -2)
    goal = (8, 6)
    circle_obstacle_list = [(2, 2, 1), (3, 3, 1), (4, 4, 1)]

    # Define the area
    rand_area = [-5, 15, -5, 15]
    static_obstacles = [
                        [(2, 2), (2, 8), (3, 8), (3, 3), (8, 3), (8, 2)],
                        [(6, 6), (7, 6), (7, 7), (6, 7)]
                        ]
    rrt_star = RRTStar(start, goal, circle_obstacle_list, static_obstacles, rand_area)
    path = rrt_star.plan()
    if path is None:
        print("Cannot find path")
    else:
        print("Found path")
        print(path)
        rrt_star.draw_graph()
        plt.plot([x for (x, y) in path], [y for (x, y) in path], color='blue', linewidth=3)
        plt.show()

if __name__ == '__main__':
    main()
