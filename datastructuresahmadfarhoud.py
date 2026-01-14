"""
Data Structures and Algorithms - All 10 Assignments
Student Name: Ahmad Farhoud
Student ID: 2210213543

Assignment List:
1. Sparse Matrix Conversion
2. Tower of Hanoi
3. Shunting Yard Algorithm (Infix to Postfix)
4. Max & Min Heap Implementation
5. Doubly Linked List Operations
6. Circular Linked List Operations
7. Array Insert & Delete Operations
8. Graph Clustering from Array
9. DFS and BFS Implementation
10. Tree Creation Using Array
"""

from collections import deque, defaultdict

# ============================================================================
# ASSIGNMENT 1: SPARSE MATRIX CONVERSION
# ============================================================================

class SparseMatrix:
    def init(self, matrix):
        self.rows = len(matrix)
        self.cols = len(matrix[0]) if matrix else 0
        self.sparse_form = []
        self.convert_to_sparse(matrix)
    
    def convert_to_sparse(self, matrix):
        """Convert regular matrix to sparse format"""
        for i in range(self.rows):
            for j in range(self.cols):
                if matrix[i][j] != 0:
                    self.sparse_form.append((i, j, matrix[i][j]))
    
    def display_sparse(self):
        """Display sparse matrix representation"""
        print(f"Sparse Matrix ({self.rows}x{self.cols}):")
        print("Row\tCol\tValue")
        for row, col, val in self.sparse_form:
            print(f"{row}\t{col}\t{val}")
    
    def to_dense(self):
        """Convert sparse matrix back to dense format"""
        dense = [[0] * self.cols for _ in range(self.rows)]
        for row, col, val in self.sparse_form:
            dense[row][col] = val
        return dense
    
    def transpose(self):
        """Transpose the sparse matrix"""
        transposed = [(col, row, val) for row, col, val in self.sparse_form]
        transposed.sort(key=lambda x: (x[0], x[1]))
        
        result = SparseMatrix([[]])
        result.rows = self.cols
        result.cols = self.rows
        result.sparse_form = transposed
        return result


# ============================================================================
# ASSIGNMENT 2: TOWER OF HANOI
# ============================================================================

def tower_of_hanoi(n, source, destination, auxiliary, moves=None):
    """Solve Tower of Hanoi puzzle recursively"""
    if moves is None:
        moves = []
    
    if n == 1:
        move = f"Move disk 1 from {source} to {destination}"
        moves.append(move)
        print(move)
        return moves
    
    tower_of_hanoi(n-1, source, auxiliary, destination, moves)
    
    move = f"Move disk {n} from {source} to {destination}"
    moves.append(move)
    print(move)
    
    tower_of_hanoi(n-1, auxiliary, destination, source, moves)
    
    return moves


def count_hanoi_moves(n):
    """Calculate minimum number of moves required"""
    return 2**n - 1


# ============================================================================
# ASSIGNMENT 3: SHUNTING YARD ALGORITHM (INFIX TO POSTFIX)
# ============================================================================