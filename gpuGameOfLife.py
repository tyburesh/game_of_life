# Conway's game of life using PyCUDA
# This program is designed to be run on GPU

# imports
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.tools import DeviceData
from pycuda.compiler import SourceModule

# NOTE
# I have assumed a single block of (MATRIX_SIZE x MATRIX_SIZE) threads
# MATRIX_SIZE squared therefore cannot exceed max_threads
# max_threads is a number specific to each device

# constants
MATRIX_SIZE = 100 # board dimensions
INTERVAL = 250 # in milliseconds
kernel = """
	__global__ void lifeStep(float *board)
	{

		int size = %(MATRIX_SIZE)s
		int x = threadIdx.x		// current cell x value
		int y = threadIdx.y		// current cell y value

		int num = board[((y - 1) * size) mod size + x] +				// above
			board[((y - 1) * size) mod size + (x - 1) mod size] +		// above and left
			board[((y - 1) * size) mod size + (x + 1) mod size] +		// above and right
			board[(y * size) + (x - 1) mod size] +						// left
			board[(y * size) + (x + 1) mod size] +						// right
			board[((y + 1) * size) mod size + x] +						// below
			board[((y + 1) * size) mod size + (x - 1) mod size] +		// below and left
			board[((y + 1) * size) mod size + (x + 1) mod size]			// below and right

		int liveAnd2 = board[y * size + x] && (num == 2)
		int liveAnd3 = board[y * size + x] && (num == 3)
		int deadAnd3 = !(board[y * size + x]) && (num == 3)

		// write the new value back to the board
		board[y * size + x] = liveAnd2 || liveAnd3 || deadAnd3
	}
"""

class Game:
	def __init__(self, size):
		self.size = size
		self.initialize_board(size)
		self.initialize_kernel()
		self.run()

	# Each cell is randomly set to either 1 (live) or 0 (dead)
	def initialize_board(self, size):
		self.board = np.random.randint(2, size = (self.size, self.size)).astype(np.float32)

	# Incomplete
	# Will utilize pycuda to parallelize computation
	def initialize_kernel(self):
		kernel_template = """
		"""

		# Transfer CPU memory to GPU memory
		self.board_gpu = gpuarray.to_gpu(self.board)

		# Get kernel code and specify matrix_size
		self.kernel_code = kernel_template % {
			'MATRIX_SIZE': self.size
		}

		# Compile kernel code
		self.mod = SourceModule(kernel_template)

		# Get kernel function from compiled module
		self.game = mod.get_function("")
		pass

	# Update each cell of the grid
	# Any live cell with less than two live neighbors dies
	# Any live cell with two or three live neighbors lives
	# Any live cell with four or more live neighbors dies
	# Any dead cell with three neighbors becomes a live cell
	def step(self, frame, img):
		self.next_board = self.board.copy()
		for i in range(self.size):
			for j in range(self.size):

				# Game of life classically takes place on an infinite grid
				# I've used a toirodal geometry for the problem
				# The matrix wraps from top to bottom and from left to right
				num = int(self.board[(i-1)%self.size][(j-1)%self.size] + \
					self.board[(i+1)%self.size][(j+1)%self.size] + \
					self.board[(i-1)%self.size][(j+1)%self.size] + \
					self.board[(i+1)%self.size][(j-1)%self.size] + \
					self.board[i][(j-1)%self.size] + \
					self.board[i][(j+1)%self.size] + \
					self.board[(i-1)%self.size][j] + \
					self.board[(i+1)%self.size][j])

				# Live cell
				if self.board[i][j] == 1:
					if (num < 2 or num > 3):
						self.next_board[i][j] = 0

				# Dead cell
				else:
					if (num == 3):
						self.next_board[i][j] = 1

		# Update animation and save updated board
		img.set_data(self.next_board)
		self.board[:] = self.next_board[:]
		return img

	# Main driver function
	# Setup animation and begin game
	def run(self):
		fig, ax = plt.subplots()
		img = ax.imshow(self.board, interpolation='nearest') 
		ani = animation.FuncAnimation(fig, self.step, fargs = (img,), frames = None, interval = INTERVAL) 
		plt.show()


if __name__ == '__main__':
	Game(MATRIX_SIZE)