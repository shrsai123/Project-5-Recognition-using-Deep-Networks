# Design Your Own Experiment: MNIST CNN Architecture Search

## Plan

I will evaluate how changing CNN architecture and training choices affects MNIST accuracy and training time. The experiment uses the same fixed training subset and same fixed test subset for every run so the comparisons are fair.

Dimensions:

- `conv1_filters`: 6, 10, 16, 24
- `conv2_filters`: 12, 20, 32, 48
- `kernel_size`: 3, 5, 7
- `dropout`: 0.0, 0.25, 0.5, 0.65
- `hidden_nodes`: 24, 50, 100, 160
- `batch_size`: 64, 128, 256, 512

Search strategy:

- Start with a baseline close to the original CNN.
- Change one dimension at a time around the baseline.
- Fill the rest of the 50-run budget with randomized combinations of the same options.

Metrics:

- Test accuracy and test error
- Training accuracy
- Parameter count
- Total training time
- Seconds per epoch

## Hypotheses

1. More convolution filters should improve accuracy up to a point, but increase training time.
2. Kernel size 5 should be strong because it matches the original network; kernel size 3 may miss wider stroke patterns, while 7 may add cost.
3. Moderate dropout should generalize better than no dropout, but very high dropout should slow learning.
4. More hidden nodes should help until the classifier has enough capacity, after which training cost grows faster than accuracy.
5. Larger batch sizes should train faster per epoch, but may slightly reduce final accuracy under a fixed epoch budget.

## Execution

Run:

```powershell
py -3.11 experiment_cnn.py
```

Outputs are written to `outputs/experiment/`.
