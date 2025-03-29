# from numba import int32, float32
# from numba.experimental import jitclass
import numpy as np


class Node:
    def __init__(self, value, next):
        self.value = value
        self.next = next


class Queue:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def enqueue(self, value):
        node = Node(value, None)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self.size += 1

    def dequeue(self):
        if self.head is None:
            return None
        value = self.head.value
        self.head = self.head.next
        self.size -= 1
        return value

    def is_empty(self):
        return self.head is None

    def get_size(self):
        return self.size


# @jitclass([("value1", float32), ("value2", float32), ("next", int32)])
# class NumbaFloatIntNode:
#     def __init__(self, value1, value2, next):
#         self.value1 = value1
#         self.value2 = value2
#         self.next = next
#
#
# @jitclass([("head", int32), ("tail", int32), ("size", int32), ("max_size", int32)])
# class NumbaFloatIntQueue:
#     def __init__(self, max_size):
#         self.head = -1
#         self.tail = -1
#         self.size = 0
#         self.max_size = max_size
#         self.nodes = np.empty(max_size, dtype=NumbaFloatIntNode)  # GPU cant do dynamic memory
#
#     def enqueue(self, value1, value2):
#         # Add a new node to the rear of the queue
#         if self.size == self.max_size:
#             raise OverflowError("Queue is full")
#
#         new_node = NumbaFloatIntNode(value1, value2)
#
#         # Add the new node to the node pool
#         node_index = self.size  # The index where the new node will go
#         self.nodes[node_index] = new_node
#
#         # Link the node in the queue
#         if self.size == 0:
#             # Queue is empty, both head and tail point to the new node
#             self.head = node_index
#             self.tail = node_index
#         else:
#             # Link the previous tail node to the new node
#             self.nodes[self.tail].next = node_index
#             self.tail = node_index  # Update tail to the new node
#
#         self.size += 1
#
#     def dequeue(self):
#         if self.size == 0:
#             raise IndexError("Queue is empty")
#
#         front_index = self.head
#         front_node = self.nodes[front_index]
#         self.head = front_node.next
#
#         self.size -= 1
#         return front_node
#
#     def peek(self):
#         if self.size == 0:
#             raise IndexError("Queue is empty")
#
#         return self.nodes[self.head]
#
#     def get_tail(self):
#         if self.size == 0:
#             raise IndexError("Queue is empty")
#
#         return self.nodes[self.tail]
#
#     def is_empty(self):
#         return self.size == 0
#
#     def get_size(self):
#         return self.size
