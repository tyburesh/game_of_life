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
MATRIX_SIZE = 8 # size of square board
BLOCK_SIZE = 2 # number of blocks
N_ITERS = 100 # number of iterations

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
		self.board = np.array([[0., 1., 1., 0., 0., 1., 1., 0.],
			[1., 1., 0., 0., 1., 0., 0., 0.],
			[1., 0., 0., 0., 1., 1., 1., 0.],
			[0., 1., 0., 0., 0., 0., 1., 1.],
			[0., 1., 1., 0., 0., 1., 1., 0.],
			[1., 0., 0., 0., 1., 0., 0., 0.],
			[1., 0., 0., 0., 1., 1., 1., 0.],
			[0., 1., 0., 1., 0., 0., 1., 1.]
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

				// Matrix size - hard coded for now
				unsigned int m_size = 8;
				unsigned int num_cells = 64;

				// Column index of the element
				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
				// Row index of the element
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
				// Thread ID in the board array
				unsigned int thread_id = y * m_size + x;

				unsigned int above = (thread_id - m_size) % num_cells;
				unsigned int below = (thread_id + m_size) % num_cells;
				unsigned int left;
				if (thread_id % m_size == 0) {
					left = thread_id + m_size - 1;
				} else {
					left = thread_id - 1;
				}
				unsigned int right;
				if (thread_id % m_size == m_size - 1) {
					right = thread_id - m_size + 1;
				} else {
					right = thread_id + 1;
				}
				unsigned int above_left;
				if (thread_id % m_size == 0) {
					above_left = (thread_id - 1) % num_cells;
				} else {
					above_left = (thread_id - m_size - 1) % num_cells;
				}
				unsigned int above_right;
				if (thread_id % m_size == m_size - 1) {
					above_right = (thread_id - blockDim.x * m_size + 1) % num_cells;
				} else {
					above_right = (thread_id - m_size + 1) % num_cells;
				}
				unsigned int below_left;
				if (thread_id % m_size == 0) {
					below_left = (thread_id + blockDim.x * m_size - 1) % num_cells;
				} else {
					below_left = (thread_id + m_size - 1) % num_cells;
				}
				unsigned int below_right;
				if (thread_id % m_size == m_size - 1) {
					below_right = (thread_id + 1) % num_cells;
				} else {
					below_right = (thread_id + m_size + 1) % num_cells;
				}

				// Game of life classically takes place on an infinite grid
				// I've used a toirodal geometry for the problem
				// The matrix wraps from top to bottom and from left to right
				unsigned int num = board[above] + board[below] + board[left] + board[right] +
					board[above_left] + board[above_right] + board[below_left] + board[below_right];

				//printf("Thread_ID = %u 		Board Value = %f 		Num = %u\\n", thread_id, board[thread_id], num);

				// Live cell with 2 neighbors
				unsigned int live_and2 = board[thread_id] && (num == 2);

				// Live cell with 3 neighbors
				unsigned int live_and3 = board[thread_id] && (num == 3);

				// Dead cell with 3 neighbors
				unsigned int dead_and3 = !(board[thread_id]) && (num == 3);

				// write the new value back to the board
				board2[thread_id] = live_and2 || live_and3 || dead_and3;
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

		i = 0
		while i < N_ITERS: 
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

			self.board_gpu, self.next_board = self.next_board, self.board_gpu
			i += 1

		print('Final board: \n', self.board_gpu)

if __name__ == '__main__':
	Game(MATRIX_SIZE, N_ITERS, BLOCK_SIZE)