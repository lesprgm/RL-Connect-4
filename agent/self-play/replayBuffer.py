from collections import deque;
import random;

class ReplayBuffer:
    def __init__(self, maxSize):
        self.buffer = deque(maxlen = maxSize);
    
    #Adds a transition to the replay buffer 
    def add(self, transition):
        self.buffer.append(transition);
    #This samples from the replay buffer, returning a batvch of transitions.
    def sample(self, batchSize):
        bufferSize = len(self.buffer);
        #In case the buffer is smaller than the batch size, we return a batch of the size of the buffer.
        batchSize = min(batchSize, bufferSize);
        return random.sample(self.buffer, batchSize);
    
    def size(self):
        return len(self.buffer)