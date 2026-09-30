import unittest

from cognition_slm.tokenizer import BOS_ID, BYTE_OFFSET, EOS_ID, PAD_ID, ByteTokenizer, VOCAB_SIZE


class TokenizerTests(unittest.TestCase):
    def test_unsupported_vocabulary_sizes_rejected(self):
        for size in (258, 260, 512):
            with self.subTest(vocab_size=size), self.assertRaisesRegex(ValueError, "exactly 259"):
                ByteTokenizer(vocab_size=size)

    def test_round_trip_unicode(self):
        tokenizer = ByteTokenizer()
        encoded = tokenizer.encode("print('café')")
        self.assertEqual(encoded[0], BOS_ID)
        self.assertEqual(encoded[-1], EOS_ID)
        self.assertEqual(tokenizer.decode(encoded), "print('café')")

    def test_a_character_cut_off_at_the_end_is_dropped_not_replaced(self):
        tokenizer = ByteTokenizer()
        # A length stop can land inside a character; 17 bytes end one byte into the sixth one.
        ids = tokenizer.encode("东京是日本的首都", add_bos=False, add_eos=False)
        self.assertEqual(tokenizer.decode(ids[:17]), "东京是日本")
        self.assertEqual(tokenizer.decode(ids[:17] + [EOS_ID]), "东京是日本")
        # Bytes that can never start a character still show as U+FFFD.
        self.assertEqual(tokenizer.decode(tokenizer.encode("ab", add_eos=False) + [BYTE_OFFSET + 0xFF]), "ab\ufffd")

    def test_vocab_and_padding_ids_are_stable(self):
        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.vocab_size, VOCAB_SIZE)
        self.assertEqual(PAD_ID, 0)


if __name__ == "__main__":
    unittest.main()
