import numpy as np
from PIL import Image, ImageDraw
import collections
import heapq

data = np.array
data = np.genfromtxt("Colorado_844x480.dat", dtype= int, delimiter=None)

print(data.shape)

print("Min value in map: ", data.min())
print("Max value in map: ", data.max())

"""start = data[0:,0:1].min()"""

# draw the map from the data
im = Image.open("blank.jpg")
draw = ImageDraw.Draw(im)
for row in range(0, 480):
    for col in range(0, 844):
      color = int(data[row][col]/17)
      draw.point((col,row), fill=(color, color, color))
im.show()

#these methods come from implementation.py from the stanford resource
def from_id_width(id, width):
    return (id % width, id // width)

def draw_tile(graph, id, style, width):
    r = "."
    if 'number' in style and id in style['number']: r = "%d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1: r = ">"
        if x2 == x1 - 1: r = "<"
        if y2 == y1 + 1: r = "v"
        if y2 == y1 - 1: r = "^"
    if 'start' in style and id == style['start']: r = "A"
    if 'goal' in style and id == style['goal']: r = "Z"
    if 'path' in style and id in style['path']: r = "@"
    return r
#this method comes from implementation.py from the stanford resource
def draw_grid(graph, width=2, **style):
    for y in range(graph.height):
        for x in range(graph.width):
            print("%%-%ds" % width % draw_tile(graph, (x, y), style, width), end="")
        print()

#making a graph from the data
class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.values = data

    def in_bounds(self, id):
        (x, y) = id
        return 0 < x < self.width and 0 < y < self.height
#TODO: Figure out what is going on with the path going out of bounds - what is the filter method doing exactly?

    def neighbors(self, id):
        (x, y) = id
        results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
        if (x + y) % 2 == 0: results.reverse()  # aesthetics
        results = filter(self.in_bounds, results)
        return results

class GridWithWeights(SquareGrid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}

    def cost(self, from_node, to_node):
        (x, y) = from_node
        (x1, y1) = to_node
        return abs(self.weights[x][y]-self.weights[x1][y1])

g = GridWithWeights(844, 480)
g.weights = g.values

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        print(path)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

def heuristic(current_node, next_node):
    (x1, y1) = current_node
    (x2, y2) = next_node
    return abs(data[y1][x1] - data[y2][x2])

def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(current, next)
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far

start = (0, 100)
goal = (300, 100)
came_from, cost_so_far = a_star_search(g, start, goal)
#draw_grid(g, width=3, point_to=came_from, start=start, goal=goal)
draw_grid(g, width=3, path=reconstruct_path(came_from, start, goal))