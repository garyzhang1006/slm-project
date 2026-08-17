import unittest

from cognition_slm.tokenizer import BOS_ID, EOS_ID, PAD_ID, ByteTokenizer, VOCAB_SIZE


class TokenizerTests(unittest.TestCase):
    def test_round_trip_unicode(self):
        tokenizer = ByteTokenizer()
        encoded = tokenizer.encode("print('café')")
        self.assertEqual(encoded[0], BOS_ID)
        self.assertEqual(encoded[-1], EOS_ID)
        self.assertEqual(tokenizer.decode(encoded), "print('café')")

    def test_vocab_and_padding_ids_are_stable(self):
        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.vocab_size, VOCAB_SIZE)
        self.assertEqual(PAD_ID, 0)


if __name__ == "__main__":
    unittest.main()
