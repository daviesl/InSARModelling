from collections import deque
import numpy as np

class Counter(object):
    def append_if(self, queue, x, y):
        """Append to the queue only if in bounds of the grid and the cell value is 1."""
        if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1] and self.grid[x][y] == 1 and self.unqueued[x][y] == 1:
            queue.append((x, y))
            self.unqueued[x][y]=0

    def mark_neighbors(self, row, col, mark_with):
        """Mark all the cells in the current island with value = mark_with. Breadth-first search."""
        queue = deque()

        queue.append((row, col))
        while queue:
            x, y = queue.pop()
            self.grid[x][y] = mark_with
            self.append_if(queue, x - 1, y)
            self.append_if(queue, x, y - 1)
            self.append_if(queue, x + 1, y)
            self.append_if(queue, x, y + 1)
            self.append_if(queue, x - 1, y - 1)
            self.append_if(queue, x - 1, y + 1)
            self.append_if(queue, x + 1, y - 1)
            self.append_if(queue, x + 1, y + 1)
            
            # 1|2|3
            # -+-+-
            # 8|0|4
            # -+-+-
            # 7|6|5
#            if t==0:
#                self.append_if(queue, x - 1, y,8)
#                self.append_if(queue, x, y - 1,2)
#                self.append_if(queue, x + 1, y,4)
#                self.append_if(queue, x, y + 1,6)
#                self.append_if(queue, x - 1, y - 1,1)
#                self.append_if(queue, x - 1, y + 1,7)
#                self.append_if(queue, x + 1, y - 1,3)
#                self.append_if(queue, x + 1, y + 1,5)
#            elif t==1:
#                self.append_if(queue, x + 1, y + 1,5)
#                self.append_if(queue, x - 1, y,8)
#                self.append_if(queue, x - 1, y - 1,1)
#                self.append_if(queue, x, y - 1,2)
#                self.append_if(queue, x - 1, y + 1,7)
#            elif t==2:
#                self.append_if(queue, x - 1, y - 1,1)
#                self.append_if(queue, x, y - 1,2)
#                self.append_if(queue, x - 1, y + 1,7)
#            elif t==3:
#                self.append_if(queue, x - 1, y - 1,1)
#                self.append_if(queue, x, y - 1,2)
#                self.append_if(queue, x - 1, y + 1,7)
#                self.append_if(queue, x + 1, y,4)
#                self.append_if(queue, x + 1, y - 1,3)
#            elif t==4:
#                self.append_if(queue, x - 1, y + 1,7)
#                self.append_if(queue, x + 1, y,4)
#                self.append_if(queue, x + 1, y - 1,3)
#            elif t==5:
#                self.append_if(queue, x - 1, y + 1,7)
#                self.append_if(queue, x + 1, y,4)
#                self.append_if(queue, x + 1, y - 1,3)
#                self.append_if(queue, x, y + 1,6)
#                self.append_if(queue, x + 1, y + 1,5)
#            elif t==6:
#                self.append_if(queue, x + 1, y - 1,3)
#                self.append_if(queue, x, y + 1,6)
#                self.append_if(queue, x + 1, y + 1,5)
#            elif t==7:
#                self.append_if(queue, x + 1, y - 1,3)
#                self.append_if(queue, x, y + 1,6)
#                self.append_if(queue, x + 1, y + 1,5)
#                self.append_if(queue, x - 1, y,8)
#                self.append_if(queue, x - 1, y - 1,1)
#            elif t==8:
#                self.append_if(queue, x + 1, y + 1,5)
#                self.append_if(queue, x - 1, y,8)
#                self.append_if(queue, x - 1, y - 1,1)
                

    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """

        if grid.shape[0] == 0 or grid.shape[1] == 0:
            return 0

        self.grid = grid

        row_length = grid.shape[0]
        col_length = grid.shape[1]

        island_counter = 0

                

    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """

        if grid.shape[0] == 0 or grid.shape[1] == 0:
            return 0

        self.grid = grid
	self.unqueued = np.full_like(grid,1)

        row_length = grid.shape[0]
        col_length = grid.shape[1]

        island_counter = 0

        indices = np.argwhere(self.grid == 1)
        numindices = indices.shape[0]
	i = 0
	while numindices > 0:
            row = indices[i][0]
            col = indices[i][1]
            i += 1
            numindices -= 1
            if self.grid[row][col] == 1:
                # found an island
                island_counter += 1
                self.mark_neighbors(row, col, island_counter + 1)
                indices = np.argwhere(self.grid == 1)
                numindices = indices.shape[0]
                i = 0

	#indices = np.transpose(np.nonzero(self.grid))
        #for i in xrange(indices.shape[0]):
        #        row = indices[i][0]
        #        col = indices[i][1]
        #        if self.grid[row][col] == 1:
        #            # found an island
        #            island_counter += 1
        #            self.mark_neighbors(row, col, island_counter + 1)

        #for row in range(row_length):
        #    for col in range(col_length):
        #        if self.grid[row][col] == 1:
        #            # found an island
        #            island_counter += 1
        #            self.mark_neighbors(row, col, island_counter + 1)

        return island_counter

    def maxIsland(self):
        from scipy.stats import mode
        m = mode(self.grid,axis=None)
        return m[0][0]
