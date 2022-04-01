import math

def get_parent_nodes(t):
    """
    Return a list of parent nodes (0-indexing) of leave node t (1-indexing).
    E.g., if t = 7, then return [3, 5, 6].
    Input: 
        t, dec interger, 1-indexing.
    """
    indices = []
    binary = bin(t)[2:]
    n = len(binary)     # ceil(log_2(t))
    temp = 0
    for (i, s) in enumerate(binary):
        if s == '1':
            temp += 2**(n-1-i)
            indices.append(temp-1)
    return indices

# test
if __name__ == "__main__":
    print(get_parent_nodes(7))  # [3,5,6]