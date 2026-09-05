import unittest


try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch not installed")
class ModelTests(unittest.TestCase):
    def test_forward_shapes_and_loss(self):
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CognitionSLM

        config = ModelConfig(block_size=32, n_layer=1, n_head=2, n_embd=16)
        model = CognitionSLM(config)
        input_ids = torch.randint(3, config.vocab_size, (2, 12))
        mask = torch.ones_like(input_ids)
        pool_positions = torch.tensor([4, 7])
        output = model(
            input_ids,
            attention_mask=mask,
            pool_positions=pool_positions,
            task_labels=torch.tensor([0, 1]),
            error_labels=torch.tensor([0, 2]),
            confidence_labels=torch.tensor([1, 2]),
        )
        self.assertEqual(tuple(output.logits.shape), (2, 12, config.vocab_size))
        self.assertEqual(tuple(output.task_logits.shape), (2, len(config.task_types)))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))

    def test_answer_only_language_loss_can_ignore_prompt_tokens(self):
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CognitionSLM

        model = CognitionSLM(ModelConfig(block_size=16, n_layer=1, n_head=2, n_embd=16))
        input_ids = torch.randint(3, model.config.vocab_size, (1, 8))
        prompt_only = torch.zeros((1, 7), dtype=torch.bool)
        output = model(input_ids, lm_loss_mask=prompt_only)

        self.assertEqual(float(output.lm_loss), 0.0)

    def test_pool_position_rejects_padding(self):
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CognitionSLM

        model = CognitionSLM(ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=16))
        input_ids = torch.randint(3, model.config.vocab_size, (1, 4))
        mask = torch.tensor([[1, 1, 1, 0]])
        with self.assertRaises(ValueError):
            model(input_ids, attention_mask=mask, pool_positions=torch.tensor([3]))

    def test_modern_architecture_forward_and_weight_tying(self):
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CognitionSLM

        config = ModelConfig(
            block_size=32,
            n_layer=1,
            n_head=2,
            n_embd=16,
            architecture="modern",
        )
        model = CognitionSLM(config)
        input_ids = torch.randint(3, config.vocab_size, (2, 12))
        output = model(input_ids, pool_positions=torch.tensor([4, 7]))

        self.assertIsNone(model.position_embedding)
        self.assertIs(model.token_embedding.weight, model.lm_head.weight)
        self.assertEqual(tuple(output.logits.shape), (2, 12, config.vocab_size))
        self.assertTrue(torch.isfinite(output.loss))

    def test_attention_matches_reference_with_padding(self):
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CausalSelfAttention, _rotate_half

        for architecture in ("legacy", "modern"):
            config = ModelConfig(block_size=16, n_layer=1, n_head=2, n_embd=16,
                                 architecture=architecture, dropout=0.2)
            attention = CausalSelfAttention(config).eval()
            x = torch.randn(2, 8, 16)
            for mask in (None, torch.tensor([[1] * 8, [1] * 5 + [0] * 3])):
                qkv = attention.qkv(x).view(2, 8, 3, 2, 8)
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
                if architecture == "modern":
                    cos, sin = attention.rope_cos[:, :, :8], attention.rope_sin[:, :, :8]
                    q = q * cos + _rotate_half(q) * sin
                    k = k * cos + _rotate_half(k) * sin
                scores = (q @ k.transpose(-2, -1)) / (8 ** 0.5)
                scores = scores.masked_fill(attention.causal_mask[:8, :8], float("-inf"))
                if mask is not None:
                    scores = scores.masked_fill(mask[:, None, None, :].eq(0), float("-inf"))
                expected = torch.softmax(scores, dim=-1) @ v
                expected = attention.proj(expected.transpose(1, 2).contiguous().view(2, 8, 16))
                torch.testing.assert_close(attention(x, mask), expected, atol=1e-6, rtol=1e-5)

    def test_causality_preserves_prompt_logits_and_heads(self):
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CognitionSLM

        model = CognitionSLM(ModelConfig(block_size=16, n_layer=2, n_head=2,
                                         n_embd=16, architecture="modern")).eval()
        ids = torch.randint(3, 259, (1, 10))
        changed = ids.clone()
        changed[:, 5:] = torch.randint(3, 259, (1, 5))
        first = model(ids, pool_positions=torch.tensor([4]))
        second = model(changed, pool_positions=torch.tensor([4]))
        torch.testing.assert_close(first.logits[:, :5], second.logits[:, :5])
        for name in ("task_logits", "error_logits", "confidence_logits"):
            torch.testing.assert_close(getattr(first, name), getattr(second, name))

    def test_2048_boundary_backward(self):
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CognitionSLM

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CognitionSLM(ModelConfig(block_size=2048, n_layer=1, n_head=2,
                                         n_embd=16, architecture="modern")).to(device)
        ids = torch.randint(3, 259, (1, 2048), device=device)
        result = model(ids)
        self.assertEqual(tuple(result.logits.shape), (1, 2048, 259))
        result.loss.backward()
        self.assertTrue(torch.isfinite(model.token_embedding.weight.grad).all())
        with self.assertRaisesRegex(ValueError, "exceeds block_size 2048"):
            model(torch.cat((ids, ids[:, :1]), dim=1))

    def test_checkpointing_preserves_loss_and_gradients(self):
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CognitionSLM

        config = ModelConfig(block_size=16, n_layer=2, n_head=2, n_embd=16,
                             architecture="modern", dropout=0.1)
        regular = CognitionSLM(config).train()
        checkpointed = CognitionSLM(config).train()
        checkpointed.load_state_dict(regular.state_dict(), strict=True)
        checkpointed.gradient_checkpointing = True
        ids = torch.randint(3, 259, (2, 10))
        mask = torch.tensor([[1] * 10, [1] * 7 + [0] * 3])
        torch.manual_seed(42)
        expected = regular(ids, attention_mask=mask).loss
        expected.backward()
        torch.manual_seed(42)
        actual = checkpointed(ids, attention_mask=mask).loss
        actual.backward()
        torch.testing.assert_close(actual, expected)
        for (name, parameter), (other_name, other) in zip(
            regular.named_parameters(), checkpointed.named_parameters()
        ):
            self.assertEqual(name, other_name)
            if parameter.grad is None:
                self.assertIsNone(other.grad)
            else:
                torch.testing.assert_close(other.grad, parameter.grad)

    def test_invalid_masks_and_empty_input_rejected(self):
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CognitionSLM

        model = CognitionSLM(ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=16))
        ids = torch.tensor([[1, 3, 4, 0]])
        for mask in (torch.ones(1, 3), torch.tensor([[1, 2, 0, 0]]),
                     torch.tensor([[0, 0, 0, 0]]), torch.tensor([[0, 1, 1, 1]]),
                     torch.tensor([[1, 0, 1, 0]])):
            with self.assertRaises(ValueError):
                model(ids, attention_mask=mask)
        with self.assertRaisesRegex(ValueError, "non-empty shape"):
            model(ids[:, :0])

    def test_loss_excludes_padding_and_empty_targets_can_backward(self):
        from cognition_slm.config import ModelConfig
        from cognition_slm.model import CognitionSLM

        model = CognitionSLM(ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=16))
        ids = torch.tensor([[1, 3, 0, 0]])
        result = model(ids, lm_loss_mask=torch.ones(1, 3, dtype=torch.bool))
        expected = torch.nn.functional.cross_entropy(result.logits[:, 0], ids[:, 1])
        torch.testing.assert_close(result.lm_loss, expected)
        empty = model(torch.zeros((1, 4), dtype=torch.long))
        self.assertEqual(float(empty.lm_loss), 0.0)
        empty.loss.backward()
        self.assertTrue(torch.isfinite(model.token_embedding.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
