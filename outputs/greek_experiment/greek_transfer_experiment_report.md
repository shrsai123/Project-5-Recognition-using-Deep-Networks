# Greek Transfer Learning Extension

## Plan

This extension evaluates which transfer-learning choices make the pretrained MNIST network learn alpha, beta, and gamma fastest. Each run starts from the saved MNIST network, replaces `fc2` with a 3-output linear layer, and trains on the same 27 Greek examples.

Dimensions explored:

- `trainable_scope`: `fc2_only`, `fc1_fc2`, `conv2_fc`
- `optimizer_name`: `sgd`, `adam`
- `learning_rate`: 0.001, 0.01, 0.05, 0.1
- `batch_size`: 3, 5, 9, 27
- `dropout`: 0.0, 0.25, 0.5, 0.65

Metrics:

- Whether the model reaches 100% accuracy on the 27 training examples
- Epochs needed to reach 100%
- Training time
- Trainable parameter count
- Best evaluation accuracy on all 27 examples

## Hypotheses

1. Training only `fc2` should usually be fastest and least likely to overfit, because the Greek dataset is tiny.
2. Adam should reach perfect recognition in fewer epochs than SGD, especially at smaller learning rates.
3. Very large learning rates should be unstable for the small final layer.
4. Smaller batch sizes should learn faster because they create more weight updates per epoch.
5. Dropout 0.5 should be a reasonable default, but lower dropout may fit the 27 examples faster.

## Results

Total variations evaluated: 60
Runs reaching 100% accuracy: 53
Best run: 38
Best configuration: scope=conv2_fc, optimizer=sgd, lr=0.01, batch=5, dropout=0.25
Best run epochs to target: 1
Best run time: 0.197 seconds

Top 10 runs:

| Run | Target? | Epochs | Accuracy | Time | Scope | Optimizer | LR | Batch | Dropout | Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 38 | True | 1 | 1.0000 | 0.197s | conv2_fc | sgd | 0.01 | 5 | 0.25 | 21223 |
| 47 | True | 2 | 1.0000 | 0.170s | fc2_only | sgd | 0.1 | 3 | 0.50 | 153 |
| 17 | True | 2 | 1.0000 | 0.172s | fc1_fc2 | sgd | 0.001 | 3 | 0.00 | 16203 |
| 44 | True | 2 | 1.0000 | 0.178s | fc1_fc2 | adam | 0.01 | 3 | 0.50 | 16203 |
| 22 | True | 2 | 1.0000 | 0.313s | conv2_fc | adam | 0.001 | 3 | 0.00 | 21223 |
| 55 | True | 2 | 1.0000 | 0.453s | conv2_fc | sgd | 0.01 | 9 | 0.65 | 21223 |
| 25 | True | 2 | 1.0000 | 0.520s | conv2_fc | sgd | 0.05 | 9 | 0.00 | 21223 |
| 26 | True | 3 | 1.0000 | 0.217s | fc1_fc2 | adam | 0.001 | 27 | 0.00 | 16203 |
| 46 | True | 3 | 1.0000 | 0.229s | fc2_only | adam | 0.05 | 9 | 0.00 | 153 |
| 21 | True | 3 | 1.0000 | 0.230s | fc1_fc2 | sgd | 0.05 | 9 | 0.25 | 16203 |

## Dimension Averages

- `trainable_scope` success rate: conv2_fc: 0.80, fc1_fc2: 0.95, fc2_only: 0.88
- `trainable_scope` mean epochs among successful runs: conv2_fc: 5.92, fc1_fc2: 6.80, fc2_only: 7.00
- `optimizer_name` success rate: adam: 0.86, sgd: 0.89
- `optimizer_name` mean epochs among successful runs: adam: 5.05, sgd: 7.59
- `learning_rate` success rate: 0.001: 0.84, 0.01: 0.95, 0.05: 0.91, 0.1: 0.75
- `learning_rate` mean epochs among successful runs: 0.001: 9.50, 0.01: 5.48, 0.05: 4.60, 0.1: 6.83
- `batch_size` success rate: 3: 0.75, 5: 0.95, 9: 1.00, 27: 0.85
- `batch_size` mean epochs among successful runs: 3: 6.25, 5: 4.57, 9: 6.56, 27: 11.27
- `dropout` success rate: 0.0: 1.00, 0.25: 1.00, 0.5: 0.95, 0.65: 0.60
- `dropout` mean epochs among successful runs: 0.0: 5.42, 0.25: 5.00, 0.5: 8.00, 0.65: 7.33

## Discussion

The fastest average trainable scope among successful runs was `conv2_fc`. This directly tests whether the original frozen-feature transfer strategy is enough or whether unfreezing deeper layers helps.

The fastest average optimizer among successful runs was `adam`. This shows which optimizer adapted the replacement classifier most efficiently on the tiny 27-image dataset.

The best average successful batch size was `5`. Smaller batches usually perform more updates per epoch, while larger batches make each epoch more stable but can need more epochs.

The fastest average successful dropout value was `0.25`. This helps decide whether the original MNIST dropout setting is still appropriate when only a few Greek examples are available.

Because the dataset has only 27 examples, these results measure memorization and transfer efficiency rather than true generalization. The useful extension is that it compares how quickly each transfer strategy can adapt the pretrained network before testing on custom handwritten Greek symbols.
