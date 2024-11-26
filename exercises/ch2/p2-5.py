#Code explanation from github
import heapq

# classifire code style
class Node:
    def __init__(self, symbol, prob):
        self.symbol = symbol
        self.prob = prob
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.prob < other.prob

# huffman tree
def build_huffman_tree(symbols_probs):
    heap = [Node(symbol, prob) for symbol, prob in symbols_probs]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.prob + node2.prob)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]

# tabel of huffman code
def generate_huffman_codes(node, current_code="", codes={}):
    if node is not None:
        if node.symbol is not None:
            codes[node.symbol] = current_code
        generate_huffman_codes(node.left, current_code + "0", codes)
        generate_huffman_codes(node.right, current_code + "1", codes)
    return codes

# encode
def encode(data, codes):
    return ''.join(codes[symbol] for symbol in data)

# decode
def decode(encoded_data, root):
    decoded_output = []
    current_node = root

    for bit in encoded_data:
        current_node = current_node.left if bit == '0' else current_node.right
        if current_node.symbol is not None:
            decoded_output.append(current_node.symbol)
            current_node = root

    return ''.join(decoded_output)

# example
symbols_probs = [('A', 0.5), ('B', 0.25), ('C', 0.15), ('D', 0.1)]
root = build_huffman_tree(symbols_probs)

# huffman code table
codes = generate_huffman_codes(root)

print("Huffman Codes:", codes)

data = "ABCD"
encoded_data = encode(data, codes)
print("Encoded Data:", encoded_data)

decoded_data = decode(encoded_data, root)
print("Decoded Data:", decoded_data)
