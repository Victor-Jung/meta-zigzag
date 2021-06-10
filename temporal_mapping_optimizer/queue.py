

class Spatial_Unrolling_Queue:
    """Spatial Unrolling (SU) Queue

    Set up the max size in the contructor, when calling enqueue it will return the
    first in element if the new size is greater than the max size
    """

    def __init__(self, max_size):
        self.items = []
        self.max_size = max_size

    def isEmpty(self):
        return self.items == []

    def size(self):
        return len(self.items)
    
    def print_items(self):
        print(self.items)
    
    def su_size(self):
        su_size = 1
        for elt in self.items:
            su_size *= elt[1]
        return su_size

    def enqueue(self, item):

        output_list = []    
        self.items.insert(0, item)
        su_size = self.su_size()

        while su_size > self.max_size:
            output_list.append(self.dequeue())
            su_size = self.su_size()
        
        return output_list

    def dequeue(self):
        return self.items.pop()