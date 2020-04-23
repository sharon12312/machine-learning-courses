import numpy as np

# The eight queens puzzle:
# is the problem of placing eight chess queens on an 8×8 chessboard so that no two queens threaten each other

# additional functions
def create_board_string(board):
    board_string = ''
    for i in range(N):
        for j in range(N):
            board_string += str(board[i][j])
    return board_string

def is_board_safe(board):
    board_key = create_board_string(board)

    if board_key in board_state_memory:
        print('Using cached information')
        return board_state_memory[board_key]

    row_sum = np.sum(board, axis=1)
    if len(row_sum[np.where(row_sum > 1)]) > 0:
        board_state_memory[board_key] = False
        return False

    col_sum = np.sum(board, axis=0)
    if len(col_sum[np.where(col_sum > 1)]) > 0:
        board_state_memory[board_key] = False
        return False

    diags = [board[::-1,:].diagonal(i) for i in range(-board.shape[0] + 1, board.shape[1])]
    diags.extend(board.diagonal(i) for i in range(board.shape[1] - 1, -board.shape[0], -1))

    for diag in diags:
        if np.sum(diag) > 1:
            board_state_memory[board_key] = False
            return False

    board_state_memory[board_key] = True
    return True

# an helper recursive function
def place_queen(board, column):
    if column >= N:
        return True

    for row in range(N):
        board[row][column] = 1

        safe = False
        if is_board_safe(board):
            safe = place_queen(board, column + 1)

        if not safe:
            board[row][column] = 0
        else:
            break

    return safe


# initial variables
N = 8
board_state_memory = {}
board = np.zeros((N, N), np.int8)

# print initial board
print(create_board_string(board))

board_copy = board.copy()
board_copy[0, 1] = 1
print(create_board_string(board_copy))

# check is_safe() function in a row
board_copy = board.copy()
board_copy[0][0] = 1
board_copy[0][3] = 1

print(board_copy)
print(is_board_safe(board_copy))

# check is_safe() function in a columns
board_copy = board.copy()
board_copy[1][0] = 1
board_copy[2][0] = 1

print(board_copy)
print(is_board_safe(board_copy))

# check is_safe() function in a diagonal position
board_copy = board.copy()
board_copy[1][0] = 1
board_copy[2][3] = 1
board_copy[0][1] = 1

print(board_copy)
print(is_board_safe(board_copy))

print(board_state_memory)

# initial board
board = np.zeros((N, N), np.int8)
placed = place_queen(board, 0)
print(placed)

print(board)
print(board_state_memory)

# verify cache mechanism
board = np.zeros((N, N), np.int8)
placed = place_queen(board, 0)
print(placed)