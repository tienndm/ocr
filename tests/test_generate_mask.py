import unittest
import torch
from inference.trainer import generate_square_subsequent_mask

class TestGenerateSquareSubsequentMask(unittest.TestCase):
    def test_mask_values(self):
        sz = 4
        mask = generate_square_subsequent_mask(sz)
        # Expected: zeros on and below diagonal, -inf above diagonal.
        for i in range(sz):
            for j in range(sz):
                if j > i:
                    self.assertEqual(mask[i, j].item(), float("-inf"))
                else:
                    self.assertEqual(mask[i, j].item(), 0.0)

    def test_mask_shape(self):
        sz = 5
        mask = generate_square_subsequent_mask(sz)
        self.assertEqual(mask.shape, (sz, sz))

if __name__ == '__main__':
    unittest.main()
