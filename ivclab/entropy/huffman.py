import numpy as np
import constriction

class HuffmanCoder:

    def __init__(self, lower_bound=0):
        self.lower_bound = lower_bound

        self.probs = None
        self.encoder_codebook = None
        self.decoder_codebook = None

    def train(self, probs):
        self.probs = probs
        self.encoder_codebook = constriction.symbol.huffman.EncoderHuffmanTree(probs)
        self.decoder_codebook = constriction.symbol.huffman.DecoderHuffmanTree(probs)

    def encode(self, message):
        if self.encoder_codebook is None:
            raise ValueError("Huffman codec must be trained first with the probabilities: call '.train(probs)'")
        
        encoder = constriction.symbol.QueueEncoder()
        for symbol in message:
            encoder.encode_symbol(symbol - self.lower_bound, self.encoder_codebook)
        
        compressed, bitrate = encoder.get_compressed()
        return np.asarray(compressed), bitrate
    
    def decode(self, compressed, message_length):
        if self.decoder_codebook is None:
            raise ValueError("Huffman codec must be trained first with the probabilities: call '.train(probs)'")
        
        decoder = constriction.symbol.QueueDecoder(compressed)
        decoded = []
        for i in range(message_length):
            symbol = decoder.decode_symbol(self.decoder_codebook)
            decoded.append(symbol + self.lower_bound)
        
        return np.asarray(decoded)
    
if __name__ == '__main__':
    huffman = HuffmanCoder()
    huffman.train(np.array([0.3, 0.2, 0.4, 0.1], dtype=np.float32))

    message = np.asarray([1, 3, 2, 3, 0, 1, 3, 0, 2, 1, 1, 3, 3, 1, 2, 0, 1, 3, 1])
    compressed, bitrate = huffman.encode(message)

    print(f"Compressed: {compressed}")
    print(f"Bitrate: {bitrate}")

    decoded_message = huffman.decode(compressed, message_length=len(message))
    print( decoded_message == message )