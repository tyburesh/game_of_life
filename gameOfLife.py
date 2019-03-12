
# Conway's game of life using PyCUDA
# This program is designed to be run on GPU

# imports
#import pycuda.autoinit
#import pycuda.driver as drv
import numpy as np
import matplotlib.pyplot as plt
import random

# constants
BSIZE = 50 # board dimensions
RUNTIME = 5 # in seconds

class Game:
	def __init__(self, size):
		self.size = size
		self.initialize_board(size)
		self.reveal()

	def initialize_board(self, size):
		self.board = np.zeros((size, size))
		for x in range(size):
			for y in range(size):
				num = random.randint(0, 1)
				if num == 1:
					self.board[x][y] = 1

	def reveal(self):
		plt.imshow(self.board, cmap = 'binary')
		plt.show(block = False)
		plt.pause(RUNTIME)
		plt.close()

if __name__ == '__main__':
	Game(BSIZE)