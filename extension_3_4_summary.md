# Extensions 3 and 4 Summary

## Extension 3: Fixed Gabor Filter Bank

I replaced the first convolution layer of the MNIST CNN with a fixed bank of 10 hand-designed 7x7 Gabor filters. The filters cover multiple orientations, wavelengths, and phases. The first layer was frozen by setting `requires_grad = False`, so only the second convolution layer and dense classifier were trained.

Command:

```powershell
py -3.11 mnist_gabor_fixed.py --epochs 5 --batch-size 256 --test-batch-size 1000
```

Result after 5 epochs:

- Training accuracy: 95.52%
- Test accuracy: 97.09%
- Test error: 2.91%
- Frozen first-layer parameters: 490
- Trainable parameters: 30,580

The fixed Gabor model did very well. It outperformed the original CNN run from this project, which reached 96.14% test accuracy after 5 epochs. This suggests that edge- and stroke-oriented filters are a strong first-layer prior for MNIST digits. The model does not need to learn those low-level filters from scratch, so the later layers can focus on combining the fixed edge responses into digit-specific features.

## Extension 4: More Sophisticated Transformer

I improved the transformer architecture by replacing the simple patch-flattening linear layer with a convolutional patch embedding block and replacing the simple CLS-only classifier with a fused CLS-plus-mean-token classifier.

Changes:

- Convolutional patch embedding: `Conv2d -> GELU -> 1x1 Conv2d`
- LayerNorm applied to patch tokens
- Norm-first transformer encoder layers with GELU activations
- Classification from both the CLS token and the average of all patch tokens
- LayerNorm and MLP classifier
- AdamW optimizer with weight decay

Command:

```powershell
py -3.11 mnist_transformer_advanced.py --epochs 5 --batch-size 256 --test-batch-size 1000
```

Result after 5 epochs:

- Training accuracy: 95.85%
- Test accuracy: 96.69%
- Test error: 3.31%
- Trainable parameters: 57,530

The advanced transformer improved over the earlier simple transformer, which reached 95.98% test accuracy after 5 epochs. The gain was modest but consistent with the hypothesis that a richer patch embedding and classifier can make better use of the token sequence. I also tried overlapping patches first, but that version was too slow on CPU, so the final run uses non-overlapping 7x7 patches while keeping the improved embedding and classification layers.
