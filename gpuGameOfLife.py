# Conway's game of life using PyCUDA
# This program is designed to be run on GPU

# imports
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.tools import DeviceData
from pycuda.compiler import SourceModule

# constants
MATRIX_SIZE = 4 # size of square board
BLOCK_SIZE = 2 # number of blocks
N_ITERS = 1 # number of iterations

class Game:
	def __init__(self, matrix, iters, block):
		self.size = matrix
		self.n_iters = iters
		self.n_blocks = matrix // block
		self.n_threads = block
		self.initialize_board()
		self.initialize_kernel()
		self.run()

	# Each cell is randomly set to either 1 (live) or 0 (dead)
	def initialize_board(self):
		#self.board = np.random.randint(2, size = (self.size, self.size)).astype(np.float32)
		self.board = np.array([[0., 1., 2., 3.],
			[4., 5., 6., 7.],
			[8., 9., 10., 11.],
			[12., 13., 14., 15.]
		]).astype(np.float32)

	# Incomplete
	# Will utilize pycuda to parallelize computation
	def initialize_kernel(self):
		self.kernel_code = """
			// Update each cell of the grid
			// Any live cell with less than two live neighbors dies
			// Any live cell with two or three live neighbors lives
			// Any live cell with four or more live neighbors dies
			// Any dead cell with three neighbors becomes a live cell
			__global__ void life_step(float *board, float *board2)
			{

				// Matrix size hard coded for now
				unsigned int m_size = 4;

				// Column index of the element
				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
				// Row index of the element
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
				// Thread ID in the board array
				unsigned int thread_id = y * m_size + x;

				printf("Thread_ID = %u 		Board Value = %f\\n", thread_id, board[thread_id]);

				// Game of life classically takes place on an infinite grid
				// I've used a toirodal geometry for the problem
				// The matrix wraps from top to bottom and from left to right
				//unsigned int num = 0;

				// Live cell with 2 neighbors
				//unsigned int live_and2 = board[thread_id] && (num == 2);

				// Live cell with 3 neighbors
				//unsigned int live_and3 = board[thread_id] && (num == 3);

				// Dead cell with 3 neighbors
				//unsigned int dead_and3 = !(board[thread_id]) && (num == 3);

				// Make sure all of the threads in the block have done their computation
				//__syncthreads();

				// write the new value back to the board
				//board2[thread_id] = live_and2 || live_and3 || dead_and3;
			}
		"""

		# Transfer CPU memory to GPU memory
		self.board_gpu = gpuarray.to_gpu(self.board)
		self.next_board = gpuarray.empty((self.size, self.size), np.float32)

		#self.kernel = self.kernel_code.format(s = self.size)
		#self.kernel = self.kernel_code % {'MATRIX_SIZE': self.size}
		self.kernel = self.kernel_code

		# Compile kernel code
		self.mod = SourceModule(self.kernel)

		# Get kernel function from compiled module
		self.game = self.mod.get_function('life_step')

	# Main driver function
	def run(self):

		#print('Board_gpu before the call to lifeStep: ', self.board_gpu)

		# Call the kernel on our board
		self.game(
			# input
			self.board_gpu,
			# output
			self.next_board,
			# grid of multiple blocks
			grid = (self.n_blocks, self.n_blocks, 1),
			# one block of MATRIX_SIZE x MATRIX_SIZE threads
			block = (self.n_threads, self.n_threads, 1),
			)

		#print('Next_board after the call to lifeStep: ', self.next_board)


if __name__ == '__main__':
	Game(MATRIX_SIZE, N_ITERS, BLOCK_SIZE)