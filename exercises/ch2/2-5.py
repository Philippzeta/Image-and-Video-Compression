import numpy as np
import matplotlib.pyplot as plt
from ivclab.utils import imread
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.image import three_pixels_predictor
import heapq
from collections import defaultdict


# 霍夫曼树节点定义
class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


# 构建霍夫曼树
def build_huffman_tree(pmf):
    heap = []
    for symbol, freq in enumerate(pmf):
        if freq > 0:
            node = HuffmanNode(symbol, freq)
            heapq.heappush(heap, node)

    if len(heap) == 0:
        return None

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]


# 生成霍夫曼编码字典
def generate_huffman_codes(root):
    codes = {}

    def traverse(node, current_code):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = current_code
            return
        traverse(node.left, current_code + '0')
        traverse(node.right, current_code + '1')

    traverse(root, '')
    return codes


# 读取图像并计算 PMF
lena_img = imread('../../data/lena_small.tif')
residual_image_Y, residual_image_CbCr = three_pixels_predictor(lena_img, subsample_color_channels=False)
merged_residuals = np.concatenate([residual_image_Y.ravel(), residual_image_CbCr.ravel()])

# 定义符号范围，这里假设残差在 -255 到 254 之间
symbols = np.arange(-255, 255)
pmf = stats_marg(merged_residuals, symbols)

# 归一化 PMF，确保总和为1，并添加 eps 防止零概率
pmf = pmf / np.sum(pmf) + np.finfo(float).eps

# 使用独立的霍夫曼编码生成器生成编码字典
huffman_tree = build_huffman_tree(pmf)
if huffman_tree is None:
    print("PMF 中没有有效的符号。")
    codewords = {}
else:
    codewords = generate_huffman_codes(huffman_tree)

# 计算每个符号的编码长度
Codelengths = [len(codewords[symbol]) if symbol in codewords else 0 for symbol in symbols]

# 绘制编码长度图
plt.figure(figsize=(12, 6))
plt.plot(symbols, Codelengths, 'o-', linewidth=1, markersize=2)
plt.title('Length of Codeword')
plt.xlabel('Symbol')
plt.ylabel('Codeword Length')
plt.grid(True)
plt.show()

# 打印编码信息
valid_lengths = [length for length in Codelengths if length > 0]
num_codewords = len(valid_lengths)
max_codeword_length = max(valid_lengths) if valid_lengths else 0
min_codeword_length = min(valid_lengths) if valid_lengths else 0

print(f"Number of codewords: {num_codewords}")
print(f"Max. codeword length: {max_codeword_length}")
print(f"Min. codeword length: {min_codeword_length}")

# 使用 constriction 库进行编码和解码

# 初始化并训练 HuffmanCoder
huffman = HuffmanCoder()
huffman.train(pmf)

# 示例消息（符号范围应与 PMF 对应）
# 选择一些有编码的符号进行编码
# 这里我们选择符号范围中一些有非零概率的符号
# 例如，选择从 symbols 中的前 20 个非零符号
non_zero_symbols = symbols[pmf > 0]
if len(non_zero_symbols) < 20:
    message = non_zero_symbols
else:
    message = non_zero_symbols[:20]  # 取前20个有非零概率的符号
message = np.asarray(message, dtype=np.int32)

# 编码消息
compressed, bitrate = huffman.encode(message)
print(f"Compressed: {compressed}")
print(f"Bitrate: {bitrate}")

# 解码消息
decoded_message = huffman.decode(compressed, message_length=len(message))
print(f"Decoding correct: {np.array_equal(decoded_message, message)}")
