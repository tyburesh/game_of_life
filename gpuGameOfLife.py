# Conway's game of life using PyCUDA
# This program is designed to be run on GPU

# imports
import numpy as np
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
MATRIX_SIZE = 10 # board dimensions

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
			// Update each cell of the grid
			// Any live cell with less than two live neighbors dies
			// Any live cell with two or three live neighbors lives
			// Any live cell with four or more live neighbors dies
			// Any dead cell with three neighbors becomes a live cell
			__global__ void lifeStep(float *board, float *board2)
			{

				int size = %(MATRIX_SIZE)s;
				int m = size * size;
				int x = threadIdx.x;											// current cell x value
				int y = threadIdx.y;											// current cell y value

				// Game of life classically takes place on an infinite grid
				// I've used a toirodal geometry for the problem
				// The matrix wraps from top to bottom and from left to right
				int num = board[(((y - 1) * size) % m + x) % m] +				// above
					board[(((y - 1) * size) % m + (x - 1) % m) % m] +			// above and left
					board[(((y - 1) * size) % m + (x + 1) % m) % m] +			// above and right
					board[((y * size) + (x - 1) % m) % m] +						// left
					board[((y * size) + (x + 1) % m) % m] +						// right
					board[(((y + 1) * size) % m + x) % m] +						// below
					board[(((y + 1) * size) % m + (x - 1) % m) % m] +			// below and left
					board[(((y + 1) * size) % m + (x + 1) % m) % m];			// below and right

				// Live cell with 2 neighbors
				int liveAnd2 = board[y * size + x] && (num == 2);

				// Live cell with 3 neighbors
				int liveAnd3 = board[y * size + x] && (num == 3);

				// Dead cell with 3 neighbors
				int deadAnd3 = !(board[y * size + x]) && (num == 3);

				// write the new value back to the board
				board2[y * size + x] = liveAnd2 || liveAnd3 || deadAnd3;
			}
		"""

		# Transfer CPU memory to GPU memory
		self.board_gpu = gpuarray.to_gpu(self.board)
		self.board_gpu2 = gpuarray.to_gpu(self.board) # copy of the board for testing purposes

		# Get kernel code and specify matrix_size
		self.kernel_code = kernel_template % {
			'MATRIX_SIZE': self.size
		}

		# Compile kernel code
		self.mod = SourceModule(kernel_template)

		# Get kernel function from compiled module
		self.game = mod.get_function('lifeStep')

		print('Board before the call to lifeStep: ', self.board)

	# Main driver function
	def run(self):
		# Call the kernel on our board
		self.game(
			# input
			self.board_gpu,
			# output
			self.board_gpu2,
			# one block of MATRIX_SIZE x MATRIX_SIZE threads
			block = (MATRIX_SIZE, MATRIX_SIZE, 1),
			)

		print('Board after the call to lifeStep: ', self.board_gpu2)


if __name__ == '__main__':
	Game(MATRIX_SIZE)