import numpy as np


class Buffer:

	def __init__(self, params):
    	history_length = params.history_length
    	width = params.width
    	height = params.height
    	size = params.batch_size
    	self.buffer = np.zeros((size, width, height, history_length), dtype=np.uint8)

  	def add(self, state):
    	self.buffer[0, :, :, -1] = self.buffer[0, :, :, 1:]
    	self.buffer[0, :, :, -1] = state

  	def getState(self):
    	return self.buffer[0]

  	def getAllStates(self):
    	return self.buffer

  	def reset(self):
    	self.buffer.fill(0)