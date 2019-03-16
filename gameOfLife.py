
# Conway's game of life using PyCUDA
# This program is designed to be run on GPU

# imports
import numpy as np
import time
import matplotlib.pyplot as plt
#import pycuda.autoinit
#import pycuda.driver as drv
#import pycuda.gpuarray as gpuarray
#from pycuda.compiler import SourceModule

# constants
MATRIX_SIZE = 50 # board dimensions
PAUSETIME = 1 # in seconds
ITERATIONS = 5 # in seconds

class Game:
	def __init__(self, size):
		self.size = size
		self.initialize_board(size)
		self.initialize_kernel()
		self.run()

	def initialize_board(self, size):
		self.board = np.random.randint(1, size = (self.size, self.size)).astype(np.float32)
		self.board[5][5] = 1
		self.board[5][6] = 1
		self.board[5][7] = 1
		#self.board = np.random.randint(2, size = (self.size, self.size)).astype(np.float32)

	def initialize_kernel(self):
		kernel_template = ''
		#self.board_gpu = gpuarray.to_gpu(self.board)
		#self.kernel_code = kernel_template % {
		#	'MATRIX_SIZE': MATRIX_SIZE
		#}
		#self.mod = SourceModule(kernel_template)
		#self.game = mod.get_function("")
		pass

	# Main driver function
	def run(self):
		i = 0
		while i < ITERATIONS:
			self.reveal()
			self.step()
			i += 1

	# Update each cell of the grid
	# Any live cell with less than two live neighbors dies
	# Any live cell with two or three live neighbors lives
	# Any live cell with four or more live neighbors dies
	# Any dead cell with three neighbors becomes a live cell
	def step(self):
		self.next_board = self.board
		for i in range(self.size):
			for j in range(self.size):

				# Game of life classically takes place on an infinite grid
				# I've used a toirodal geometry for the problem
				# The matrix wraps from top to bottom and from left to right
				num = int(self.board[(i-1)%self.size][(j-1)%self.size] + \
					self.board[(i+1)%self.size][(j+1)%self.size] + \
					self.board[(i-1)%self.size][(j+1)%self.size] + \
					self.board[(i+1)%self.size][(j-1)%self.size] + \
					self.board[(i-1)%self.size][(j-1)%self.size] + \
					self.board[i][(j-1)%self.size] + \
					self.board[i][(j+1)%self.size] + \
					self.board[(i-1)%self.size][j] + \
					self.board[(i+1)%self.size][j])

				if self.board[i][j] == 1:
					if (num < 2 or num > 3):
						self.next_board[i][j] = 0

				else:
					if (num == 3):
						self.next_board[i][j] = 1

		self.board = self.next_board

	def reveal(self):
		plt.imshow(self.board, cmap = 'binary')
		plt.show(block = False)
		plt.pause(PAUSETIME)
		plt.close()

if __name__ == '__main__':
	Game(MATRIX_SIZE)



