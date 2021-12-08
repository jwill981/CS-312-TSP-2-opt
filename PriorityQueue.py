import heapq
import itertools

class PriorityQueue:
    REMOVED = '<removed-node>'
    counter = itertools.count(0, -1)

    def __init__(self):
        self.heap = []
        self.finder = {}
        pass

    # Add item to the priority queue using the given priority
    # If the item was already added, this method will update the priority
    def push(self, item, priority):
        if item in self.finder:
            self.removeItem(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.finder[item] = entry
        heapq.heappush(self.heap, entry)
        pass

    # Remove and return the item from the queue with the smallest priority
    def pop(self):
        while not self.isEmpty():
            priority, count, item = heapq.heappop(self.heap)
            if item is not self.REMOVED:
                del self.finder[item]
                return item
        raise RuntimeError("Empty queue cannot be popped")

    # Safely deletes a specific item from the queue
    # Used for updating the priority of an item
    def removeItem(self, item):
        entry = self.finder[item]
        entry[-1] = self.REMOVED

    # Returns true if nothing is in the queue, false otherwise
    def isEmpty(self):
        if self.heap:
            return False
        else:
            return True