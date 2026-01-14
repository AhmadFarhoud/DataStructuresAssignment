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
    def __init__(self, matrix):
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

class ShuntingYard:
    def __init__(self):
        self.precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
        self.right_associative = {'^'}
    
    def is_operator(self, char):
        return char in self.precedence
    
    def is_operand(self, char):
        return char.isalnum()
    
    def compare_precedence(self, op1, op2):
        if op1 not in self.precedence or op2 not in self.precedence:
            return False
        
        if op2 in self.right_associative:
            return self.precedence[op1] > self.precedence[op2]
        else:
            return self.precedence[op1] >= self.precedence[op2]
    
    def infix_to_postfix(self, expression):
        """Convert infix expression to postfix using Shunting Yard algorithm"""
        output = []
        operator_stack = []
        expression = expression.replace(" ", "")
        
        i = 0
        while i < len(expression):
            char = expression[i]
            
            if self.is_operand(char):
                num = char
                while i + 1 < len(expression) and expression[i + 1].isdigit():
                    i += 1
                    num += expression[i]
                output.append(num)
            
            elif char == '(':
                operator_stack.append(char)
            
            elif char == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                
                if operator_stack:
                    operator_stack.pop()
                else:
                    raise ValueError("Mismatched parentheses")
            
            elif self.is_operator(char):
                while (operator_stack and 
                       operator_stack[-1] != '(' and
                       self.compare_precedence(operator_stack[-1], char)):
                    output.append(operator_stack.pop())
                operator_stack.append(char)
            
            i += 1
        
        while operator_stack:
            if operator_stack[-1] in '()':
                raise ValueError("Mismatched parentheses")
            output.append(operator_stack.pop())
        
        return ' '.join(output)
    
    def evaluate_postfix(self, postfix):
        """Evaluate postfix expression"""
        stack = []
        tokens = postfix.split()
        
        for token in tokens:
            if token.isdigit():
                stack.append(int(token))
            elif self.is_operator(token):
                if len(stack) < 2:
                    raise ValueError("Invalid postfix expression")
                
                b = stack.pop()
                a = stack.pop()
                
                if token == '+':
                    result = a + b
                elif token == '-':
                    result = a - b
                elif token == '*':
                    result = a * b
                elif token == '/':
                    result = a / b
                elif token == '^':
                    result = a ** b
                
                stack.append(result)
        
        if len(stack) != 1:
            raise ValueError("Invalid postfix expression")
        
        return stack[0]


# ============================================================================
# ASSIGNMENT 4: MAX HEAP AND MIN HEAP
# ============================================================================

class MaxHeap:
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def insert(self, value):
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        parent = self.parent(i)
        if i > 0 and self.heap[i] > self.heap[parent]:
            self.swap(i, parent)
            self._heapify_up(parent)
    
    def extract_max(self):
        if len(self.heap) == 0:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        max_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return max_val
    
    def _heapify_down(self, i):
        largest = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        if left < len(self.heap) and self.heap[left] > self.heap[largest]:
            largest = left
        
        if right < len(self.heap) and self.heap[right] > self.heap[largest]:
            largest = right
        
        if largest != i:
            self.swap(i, largest)
            self._heapify_down(largest)
    
    def peek(self):
        return self.heap[0] if self.heap else None
    
    def display(self):
        print(f"Max Heap: {self.heap}")


class MinHeap:
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def insert(self, value):
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        parent = self.parent(i)
        if i > 0 and self.heap[i] < self.heap[parent]:
            self.swap(i, parent)
            self._heapify_up(parent)
    
    def extract_min(self):
        if len(self.heap) == 0:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return min_val
    
    def _heapify_down(self, i):
        smallest = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right
        
        if smallest != i:
            self.swap(i, smallest)
            self._heapify_down(smallest)
    
    def peek(self):
        return self.heap[0] if self.heap else None
    
    def display(self):
        print(f"Min Heap: {self.heap}")


# ============================================================================
# ASSIGNMENT 5: DOUBLY LINKED LIST
# ============================================================================

class DLLNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def is_empty(self):
        return self.head is None
    
    def insert_at_beginning(self, data):
        new_node = DLLNode(data)
        
        if self.is_empty():
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        
        self.size += 1
    
    def insert_at_end(self, data):
        new_node = DLLNode(data)
        
        if self.is_empty():
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        
        self.size += 1
    
    def insert_at_position(self, data, position):
        if position < 0 or position > self.size:
            print(f"Invalid position! List size is {self.size}")
            return False
        
        if position == 0:
            self.insert_at_beginning(data)
            return True
        
        if position == self.size:
            self.insert_at_end(data)
            return True
        
        new_node = DLLNode(data)
        current = self.head
        
        for _ in range(position - 1):
            current = current.next
        
        new_node.next = current.next
        new_node.prev = current
        current.next.prev = new_node
        current.next = new_node
        
        self.size += 1
        return True
    
    def delete_from_beginning(self):
        if self.is_empty():
            print("List is empty")
            return None
        
        data = self.head.data
        
        if self.head == self.tail:
            self.head = self.tail = None
        else:
            self.head = self.head.next
            self.head.prev = None
        
        self.size -= 1
        return data
    
    def delete_from_end(self):
        if self.is_empty():
            print("List is empty")
            return None
        
        data = self.tail.data
        
        if self.head == self.tail:
            self.head = self.tail = None
        else:
            self.tail = self.tail.prev
            self.tail.next = None
        
        self.size -= 1
        return data
    
    def delete_by_value(self, value):
        if self.is_empty():
            print("List is empty")
            return False
        
        current = self.head
        
        while current:
            if current.data == value:
                if current == self.head:
                    self.delete_from_beginning()
                elif current == self.tail:
                    self.delete_from_end()
                else:
                    current.prev.next = current.next
                    current.next.prev = current.prev
                    self.size -= 1
                return True
            current = current.next
        
        print(f"Value {value} not found")
        return False
    
    def traverse_forward(self):
        if self.is_empty():
            return []
        
        values = []
        current = self.head
        
        while current:
            values.append(current.data)
            current = current.next
        
        return values
    
    def traverse_backward(self):
        if self.is_empty():
            return []
        
        values = []
        current = self.tail
        
        while current:
            values.append(current.data)
            current = current.prev
        
        return values
    
    def display(self):
        if self.is_empty():
            print("List is empty")
            return
        
        print(f"Doubly Linked List (size={self.size}): ", end="")
        current = self.head
        
        print("None <- ", end="")
        while current:
            print(f"{current.data}", end="")
            if current.next:
                print(" <-> ", end="")
            current = current.next
        print(" -> None")


# ============================================================================
# ASSIGNMENT 6: CIRCULAR LINKED LIST
# ============================================================================

class CLLNode:
    def __init__(self, data):
        self.data = data
        self.next = None


class CircularLinkedList:
    def __init__(self):
        self.head = None
        self.size = 0
    
    def is_empty(self):
        return self.head is None
    
    def insert_at_beginning(self, data):
        new_node = CLLNode(data)
        
        if self.is_empty():
            new_node.next = new_node
            self.head = new_node
        else:
            current = self.head
            while current.next != self.head:
                current = current.next
            
            new_node.next = self.head
            current.next = new_node
            self.head = new_node
        
        self.size += 1
    
    def insert_at_end(self, data):
        new_node = CLLNode(data)
        
        if self.is_empty():
            new_node.next = new_node
            self.head = new_node
        else:
            current = self.head
            while current.next != self.head:
                current = current.next
            
            current.next = new_node
            new_node.next = self.head
        
        self.size += 1
    
    def insert_at_position(self, data, position):
        if position < 0 or position > self.size:
            print(f"Invalid position! List size is {self.size}")
            return False
        
        if position == 0:
            self.insert_at_beginning(data)
            return True
        
        new_node = CLLNode(data)
        current = self.head
        
        for _ in range(position - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self.size += 1
        return True
    
    def delete_from_beginning(self):
        if self.is_empty():
            print("List is empty")
            return None
        
        data = self.head.data
        
        if self.head.next == self.head:
            self.head = None
        else:
            current = self.head
            while current.next != self.head:
                current = current.next
            
            current.next = self.head.next
            self.head = self.head.next
        
        self.size -= 1
        return data
    
    def delete_from_end(self):
        if self.is_empty():
            print("List is empty")
            return None
        
        if self.head.next == self.head:
            data = self.head.data
            self.head = None
            self.size -= 1
            return data
        
        current = self.head
        while current.next.next != self.head:
            current = current.next
        
        data = current.next.data
        current.next = self.head
        self.size -= 1
        return data
    
    def delete_by_value(self, value):
        if self.is_empty():
            print("List is empty")
            return False
        
        if self.head.data == value:
            self.delete_from_beginning()
            return True
        
        current = self.head
        prev = None
        
        while True:
            if current.data == value:
                prev.next = current.next
                self.size -= 1
                return True
            
            prev = current
            current = current.next
            
            if current == self.head:
                break
        
        print(f"Value {value} not found")
        return False
    
    def traverse(self, max_iterations=None):
        if self.is_empty():
            return []
        
        values = []
        current = self.head
        iterations = max_iterations if max_iterations else self.size
        
        for _ in range(iterations):
            values.append(current.data)
            current = current.next
        
        return values
    
    def display(self):
        if self.is_empty():
            print("List is empty")
            return
        
        values = self.traverse()
        print(f"Circular Linked List (size={self.size}): ", end="")
        print(" -> ".join(map(str, values)), end="")
        print(f" -> {self.head.data} (head) -> ...")


# ============================================================================
# ASSIGNMENT 7: ARRAY OPERATIONS
# ============================================================================

class DynamicArray:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.size = 0
        self.array = [None] * capacity
    
    def __str__(self):
        return str([self.array[i] for i in range(self.size)])
    
    def is_full(self):
        return self.size == self.capacity
    
    def is_empty(self):
        return self.size == 0
    
    def resize(self):
        self.capacity *= 2
        new_array = [None] * self.capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array
    
    def insert_at_end(self, value):
        if self.is_full():
            self.resize()
        
        self.array[self.size] = value
        self.size += 1
        return True
    
    def insert_at_beginning(self, value):
        if self.is_full():
            self.resize()
        
        for i in range(self.size, 0, -1):
            self.array[i] = self.array[i - 1]
        
        self.array[0] = value
        self.size += 1
        return True
    
    def insert_at_position(self, value, position):
        if position < 0 or position > self.size:
            print(f"Invalid position! Size is {self.size}")
            return False
        
        if self.is_full():
            self.resize()
        
        for i in range(self.size, position, -1):
            self.array[i] = self.array[i - 1]
        
        self.array[position] = value
        self.size += 1
        return True
    
    def delete_from_end(self):
        if self.is_empty():
            print("Array is empty")
            return None
        
        value = self.array[self.size - 1]
        self.array[self.size - 1] = None
        self.size -= 1
        return value
    
    def delete_from_beginning(self):
        if self.is_empty():
            print("Array is empty")
            return None
        
        value = self.array[0]
        
        for i in range(self.size - 1):
            self.array[i] = self.array[i + 1]
        
        self.array[self.size - 1] = None
        self.size -= 1
        return value
    
    def delete_at_position(self, position):
        if position < 0 or position >= self.size:
            print(f"Invalid position! Size is {self.size}")
            return None
        
        value = self.array[position]
        
        for i in range(position, self.size - 1):
            self.array[i] = self.array[i + 1]
        
        self.array[self.size - 1] = None
        self.size -= 1
        return value
    
    def delete_by_value(self, value):
        position = self.search(value)
        
        if position == -1:
            print(f"Value {value} not found")
            return False
        
        self.delete_at_position(position)
        return True
    
    def search(self, value):
        for i in range(self.size):
            if self.array[i] == value:
                return i
        return -1


# ============================================================================
# ASSIGNMENT 8: GRAPH CLUSTERING
# ============================================================================

class GraphClustering:
    def __init__(self, adjacency_matrix):
        self.matrix = adjacency_matrix
        self.vertices = len(adjacency_matrix)
        self.visited = [False] * self.vertices
    
    def get_neighbors(self, vertex):
        neighbors = []
        for i in range(self.vertices):
            if self.matrix[vertex][i] == 1 and i != vertex:
                neighbors.append(i)
        return neighbors
    
    def dfs_cluster(self, start, cluster):
        self.visited[start] = True
        cluster.append(start)
        
        for neighbor in self.get_neighbors(start):
            if not self.visited[neighbor]:
                self.dfs_cluster(neighbor, cluster)
    
    def find_clusters_dfs(self):
        self.visited = [False] * self.vertices
        clusters = []
        
        for vertex in range(self.vertices):
            if not self.visited[vertex]:
                cluster = []
                self.dfs_cluster(vertex, cluster)
                clusters.append(cluster)
        
        return clusters
    
    def display_matrix(self):
        print("\nAdjacency Matrix:")
        print("  ", end="")
        for i in range(self.vertices):
            print(f"{i} ", end="")
        print()
        
        for i in range(self.vertices):
            print(f"{i} ", end="")
            for j in range(self.vertices):
                print(f"{self.matrix[i][j]} ", end="")
            print()


# ============================================================================
# ASSIGNMENT 9: DFS AND BFS
# ============================================================================

class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj_list = defaultdict(list)
    
    def add_edge(self, u, v, directed=False):
        self.adj_list[u].append(v)
        if not directed:
            self.adj_list[v].append(u)
    
    def display(self):
        print("\nAdjacency List:")
        for vertex in sorted(self.adj_list.keys()):
            print(f"{vertex} -> {self.adj_list[vertex]}")
    
    def dfs_recursive(self, start, visited=None, result=None):
        """Depth-First Search - Recursive"""
        if visited is None:
            visited = set()
            result = []
        
        visited.add(start)
        result.append(start)
        
        for neighbor in self.adj_list[start]:
            if neighbor not in visited:
                self.dfs_recursive(neighbor, visited, result)
        
        return result
    
    def bfs(self, start):
        """Breadth-First Search"""
        visited = set([start])
        queue = deque([start])
        result = []
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            for neighbor in self.adj_list[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result
    
    def bfs_shortest_path(self, start, target):
        """BFS to find shortest path"""
        if start == target:
            return [start]
        
        visited = set([start])
        queue = deque([(start, [start])])
        
        while queue:
            vertex, path = queue.popleft()
            
            for neighbor in self.adj_list[vertex]:
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None


# ============================================================================
# ASSIGNMENT 10: TREE CREATION FROM ARRAY
# ============================================================================

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BinaryTree:
    def __init__(self):
        self.root = None
    
    def create_from_array(self, arr):
        """Create complete binary tree from array"""
        if not arr:
            return None
        
        self.root = self._create_tree_recursive(arr, 0)
        return self.root
    
    def _create_tree_recursive(self, arr, index):
        if index >= len(arr) or arr[index] is None:
            return None
        
        node = TreeNode(arr[index])
        node.left = self._create_tree_recursive(arr, 2 * index + 1)
        node.right = self._create_tree_recursive(arr, 2 * index + 2)
        
        return node
    
    def inorder(self, node=None, first_call=True):
        if first_call:
            node = self.root
        
        result = []
        if node:
            result.extend(self.inorder(node.left, False))
            result.append(node.value)
            result.extend(self.inorder(node.right, False))
        return result

    def display(self, node=None, level=0, prefix="Root: ", first_call=True):
        if first_call:
            node = self.root
        
        if node:
            print(" " * (level * 4) + prefix + str(node.value))
            if node.left or node.right:
                if node.left:
                    self.display(node.left, level + 1, "L--- ", False)
                else:
                    print(" " * ((level + 1) * 4) + "L--- None")
                
                if node.right:
                    self.display(node.right, level + 1, "R--- ", False)
                else:
                    print(" " * ((level + 1) * 4) + "R--- None")


class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left:
                self._insert_recursive(node.left, value)
            else:
                node.left = TreeNode(value)
        else:
            if node.right:
                self._insert_recursive(node.right, value)
            else:
                node.right = TreeNode(value)
    
    def create_from_array(self, arr):
        for value in arr:
            self.insert(value)
        return self.root
    
    def inorder(self, node=None, first_call=True):
        if first_call:
            node = self.root
        result = []
        if node:
            result.extend(self.inorder(node.left, False))
            result.append(node.value)
            result.extend(self.inorder(node.right, False))
        return result


# ============================================================================
# MAIN PROGRAM - TEST ALL ASSIGNMENTS
# ============================================================================

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_assignment_1():
    print_header("ASSIGNMENT 1: SPARSE MATRIX")
    matrix = [[0, 0, 3, 0, 4], [0, 0, 5, 7, 0], [0, 0, 0, 0, 0], [0, 2, 6, 0, 0]]
    sparse = SparseMatrix(matrix)
    sparse.display_sparse()


def test_assignment_2():
    print_header("ASSIGNMENT 2: TOWER OF HANOI")
    tower_of_hanoi(3, 'A', 'C', 'B')


def test_assignment_3():
    print_header("ASSIGNMENT 3: SHUNTING YARD")
    sy = ShuntingYard()
    infix = "3 + 4 * 2"
    postfix = sy.infix_to_postfix(infix)
    print(f"Infix: {infix} -> Postfix: {postfix}")


def test_assignment_4():
    print_header("ASSIGNMENT 4: MAX & MIN HEAP")
    values = [10, 5, 3, 2, 15, 20, 8]
    mh = MaxHeap()
    for v in values: mh.insert(v)
    mh.display()


def test_assignment_5():
    print_header("ASSIGNMENT 5: DOUBLY LINKED LIST")
    dll = DoublyLinkedList()
    dll.insert_at_end(10)
    dll.insert_at_end(20)
    dll.display()


def test_assignment_6():
    print_header("ASSIGNMENT 6: CIRCULAR LINKED LIST")
    cll = CircularLinkedList()
    cll.insert_at_end(10)
    cll.insert_at_end(20)
    cll.display()


def test_assignment_7():
    print_header("ASSIGNMENT 7: ARRAY OPERATIONS")
    arr = DynamicArray(5)
    arr.insert_at_end(10)
    arr.insert_at_end(20)
    print(arr)


def test_assignment_8():
    print_header("ASSIGNMENT 8: GRAPH CLUSTERING")
    matrix = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    graph = GraphClustering(matrix)
    print(f"Clusters: {graph.find_clusters_dfs()}")


def test_assignment_9():
    print_header("ASSIGNMENT 9: DFS AND BFS")
    g = Graph(3)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    print(f"BFS from 0: {g.bfs(0)}")


def test_assignment_10():
    print_header("ASSIGNMENT 10: TREE FROM ARRAY")
    arr = [1, 2, 3, 4, 5]
    tree = BinaryTree()
    tree.create_from_array(arr)
    tree.display()


def main():
    print("\n" + "="*70)
    print("  DATA STRUCTURES AND ALGORITHMS - ALL 10 ASSIGNMENTS")
    print("  Student Name: Ahmad Farhoud")
    print("  Student ID: 2210213543")
    print("="*70)
    
    test_assignment_1()
    test_assignment_2()
    test_assignment_3()
    test_assignment_4()
    test_assignment_5()
    test_assignment_6()
    test_assignment_7()
    test_assignment_8()
    test_assignment_9()
    test_assignment_10()
    
    print("\n" + "="*70)
    print("  ALL ASSIGNMENTS COMPLETED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
