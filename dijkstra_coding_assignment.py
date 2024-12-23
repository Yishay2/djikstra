"""
Assignment: Solve a Maze using Dijkstra's Algorithm

Objective:
In this assignment, you will implement Dijkstra's algorithm to find the shortest path in a maze represented as an RGB image.
You will write custom data structures, including a PixelNode class and a priority queue, while ensuring efficient pathfinding.

Problem Description:
You are given an RGB image of a maze where:
- White pixels (all channels ≈ 255) represent valid paths.
- Non-white pixels (one or more channels < 255) represent obstacles.

Your goal is to find the shortest path between a start pixel and an end pixel, given their coordinates, by:
1. Converting the RGB maze image to a grayscale representation for simplicity.
2. Implementing a custom PixelNode class for maze pixels.
3. Building and utilizing a priority queue with a decrease_key operation for efficient pathfinding.
4. Using Dijkstra's algorithm to traverse the maze and determine the shortest path.

---

Tasks:

1. Grayscale Conversion
Write a function to convert the RGB maze image into grayscale using the formula:
  Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

Task:
Implement a function `convert_to_grayscale(image)` that:
- Takes the RGB image as input.
- Converts it to grayscale using the formula.
- Returns the grayscale image as a 2D array.

"""
import numpy as np
from PIL import Image


def convert_to_grayscale(image):

    """
        Convert an RGB image to grayscale.
        :param image: Input RGB image as a 3D NumPy array
        :return: Grayscale image as a 2D NumPy array
        """

    image = Image.open(image)
    numpy_data = np.array(image, dtype=np.float32)
    grayscale = 0.2989 * numpy_data[..., 0] + 0.5870 * numpy_data[..., 1] + 0.1140 * numpy_data[..., 2]
    return grayscale


convert_to_grayscale("./mazes/maze1.jpg")

"""
2. Pixel Representation
Create a class `PixelNode` to represent each pixel in the maze. It should include:
- x and y: Coordinates of the pixel.
- distance: Shortest known distance from the start pixel (initialize to infinity).
- visited: Boolean indicating whether the pixel has been processed.
- color: Optional attribute for the grayscale intensity value.
- heap_index: The index of the PixelNode in the priority queue. This is essential for efficient decrease_key operations.

Additionally:
- Implement comparison operators (e.g., __lt__) to make PixelNode objects compatible with a min-heap.

"""


class PixelNode:
    def __init__(self, x, y, distance=float('inf')):
        self.x = x
        self.y = y
        self.distance = distance
        self.visited = False
        self.heap_index = None
        self.prev = None

    def __lt__(self, other):
        return self.distance < other.distance

    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

"""
3. Priority Queue
Write a class `PriorityQueue` for managing PixelNode objects. Use a min-heap for efficient operations. Include the following methods:
- insert(node): Add a new PixelNode to the queue and update its heap_index.
- extract_min(): Remove and return the PixelNode with the smallest distance while maintaining the heap property.
- decrease_key(node, new_distance): Update the distance of an existing node and adjust its position in the heap. Update its heap_index.

Hint:
Use the heap_index attribute of PixelNode objects to keep track of their position in the heap. This allows decrease_key to access nodes efficiently.
"""


class PriorityQueue:
    def __init__(self):
        """
        Initialize an empty priority queue.
        """
        self.heap = []

    def insert(self, node: PixelNode) -> None:
        """
        Insert a PixelNode into the priority queue.
        :param node: The PixelNode to be inserted
        """
        node.heap_index = len(self.heap)
        self.heap.append(node)
        self.heapify_up(node.heap_index)

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.heap[i].heap_index = i
        self.heap[j].heap_index = j

    def heapify_up(self, idx):

        while idx > 0:
            parent_idx = (idx - 1) // 2
            if self.heap[idx] < self.heap[parent_idx]:
                self._swap(parent_idx, idx)
                idx = parent_idx
            else:
                break

    def _heapify_down(self, idx):

        size = len(self.heap)
        left_idx = 2 * idx + 1
        right_idx = 2 * idx + 2
        smallest = idx
        if left_idx < size and self.heap[left_idx] < self.heap[smallest]:
            smallest = left_idx
        if right_idx < size and self.heap[right_idx] < self.heap[smallest]:
            smallest = right_idx
        if smallest != idx:
            self._swap(idx, smallest)
            self._heapify_down(smallest)

    def extract_min(self):
        if not self.heap:
            raise Exception("Empty Priority Queue")

        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        u = self.heap.pop()
        self._heapify_down(0)
        return u

    def decrease_key(self, node, new_distance):
        """
        Update the distance of a node and re-heapify.
        :param node: The PixelNode whose distance is to be updated
        :param new_distance: The new distance value
        """
        node.distance = new_distance
        self.heapify_up(node.heap_index)


def get_all_neighbors(node, arr, image):
    neighbors = []
    x, y = node.x, node.y
    height, width = len(image), len(image[0])

    for i in range(max(0, y - 1), min(height, y + 2)):
        for j in range(max(0, x - 1), min(width, x + 2)):
            if i == y and j == x:
                continue

            if not arr[i][j].visited and image[i][j] > 200:
                neighbors.append(arr[i][j])
    return neighbors


"""
4. Dijkstra's Algorithm
Implement Dijkstra's algorithm to find the shortest path from the start pixel to the end pixel.
"""


def dijkstra(image, start, end):
    n = len(image)
    m = len(image[0])
    new_pixel_matrix = [[PixelNode(x, y) for x in range(m)] for y in range(n)]
    start_node = new_pixel_matrix[start[1]][start[0]]
    start_node.distance = 0

    heapq = PriorityQueue()
    heapq.insert(start_node)

    while len(heapq.heap) > 0:
        u = heapq.extract_min()
        u.visited = True

        if u.x == end[0] and u.y == end[1]:
            break

        neighbors = get_all_neighbors(u, new_pixel_matrix, image)
        for neighbor in neighbors:
            distance = u.distance + neighbor.distance_to(u)
            if neighbor.distance > distance:
                if neighbor.heap_index is None:
                    neighbor.distance = distance
                    neighbor.prev = u
                    heapq.insert(neighbor)
                else:
                    heapq.decrease_key(neighbor, distance)
                    neighbor.prev = u

    path = []
    current = new_pixel_matrix[end[1]][end[0]]
    while current is not None:
        path.append([current.x, current.y])
        current = current.prev

    return path[::-1]


image1 = convert_to_grayscale("./mazes/maze1.jpg")
start_point_of_maze_1 = (868, 1039)
end_point_of_maze_1 = (358, 1549)
print(dijkstra(image1, start_point_of_maze_1, end_point_of_maze_1))

# Notes:
# - Ensure all data structures are used efficiently to minimize runtime.
# - Use helper functions where necessary to keep the code modular.
# - Test your implementation with various mazes and start/end points to ensure correctness.
#
# ---
# Submission:
# Submit your completed Python file with all the required functions and classes implemented. Include comments explaining your code where necessary.
