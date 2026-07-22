# MNIST CNN Architecture Experiment

## Plan

The experiment evaluates how CNN architecture choices affect MNIST accuracy and training time. Each variation is trained on the same fixed training subset and evaluated on the same fixed test subset.

Dimensions explored:

- `conv1_filters`: 6, 10, 16, 24
- `conv2_filters`: 12, 20, 32, 48
- `kernel_size`: 3, 5, 7
- `dropout`: 0.0, 0.25, 0.5, 0.65
- `hidden_nodes`: 24, 50, 100, 160
- `batch_size`: 64, 128, 256, 512

The search starts with one-at-a-time linear variations around the baseline, then fills the remaining runs with randomized combinations. Metrics are test accuracy, test error, train accuracy, parameter count, total training time, and seconds per epoch.

## Hypotheses

1. More convolution filters should improve accuracy up to a point, but increase training time.
2. Kernel size 5 should perform well because it matches the original network; 3 may miss wider strokes and 7 may add extra cost.
3. Moderate dropout should generalize better than no dropout, while very high dropout should slow learning.
4. More hidden nodes should help until the classifier has enough capacity, after which time cost grows faster than accuracy.
5. Larger batch sizes should train faster per epoch, but may slightly reduce final accuracy for a fixed epoch count.

## Results

Total variations evaluated: 50
Best test accuracy: 0.9200 in run 31
Best architecture: conv1=16, conv2=48, kernel=7, dropout=0.65, hidden=24, batch=64
Best training time: 29.79 seconds

Top 10 runs by test accuracy:

| Run | Accuracy | Time | conv1 | conv2 | kernel | dropout | hidden | batch | params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 31 | 0.9200 | 29.79s | 16 | 48 | 7 | 0.65 | 24 | 64 | 95202 |
| 28 | 0.9195 | 28.24s | 24 | 32 | 7 | 0.65 | 100 | 64 | 196774 |
| 20 | 0.9075 | 23.63s | 16 | 12 | 7 | 0.25 | 24 | 64 | 24606 |
| 24 | 0.9055 | 23.14s | 10 | 12 | 7 | 0.50 | 50 | 64 | 36352 |
| 48 | 0.8990 | 22.28s | 6 | 20 | 7 | 0.50 | 50 | 64 | 55760 |
| 16 | 0.8980 | 25.11s | 10 | 20 | 5 | 0.50 | 50 | 64 | 54840 |
| 33 | 0.8855 | 27.46s | 24 | 32 | 3 | 0.00 | 160 | 64 | 259834 |
| 22 | 0.8695 | 22.00s | 6 | 12 | 3 | 0.00 | 160 | 64 | 96570 |
| 37 | 0.8585 | 15.82s | 24 | 20 | 3 | 0.00 | 50 | 128 | 54140 |
| 17 | 0.8490 | 16.05s | 10 | 20 | 5 | 0.50 | 50 | 128 | 54840 |

Fastest run reaching at least 90% test accuracy: run 24, 0.9055 accuracy in 23.14 seconds.

## Dimension Averages

- `conv1_filters` mean accuracy: 6: 0.6089, 10: 0.5859, 16: 0.6806, 24: 0.7214
- `conv2_filters` mean accuracy: 12: 0.5651, 20: 0.6386, 32: 0.6885, 48: 0.6958
- `kernel_size` mean accuracy: 3: 0.5886, 5: 0.6067, 7: 0.7375
- `dropout` mean accuracy: 0.0: 0.7188, 0.25: 0.4656, 0.5: 0.6777, 0.65: 0.6000
- `hidden_nodes` mean accuracy: 24: 0.6343, 50: 0.6530, 100: 0.6188, 160: 0.6065
- `batch_size` mean accuracy: 64: 0.8924, 128: 0.8017, 256: 0.5628, 512: 0.3862

## Discussion

The filter-count hypothesis was mostly supported. Mean accuracy increased from 0.5859 to 0.7214 across `conv1_filters`, and from 0.5651 to 0.6958 across `conv2_filters`, although larger models usually required more training time.

The kernel-size hypothesis was not supported. Kernel size 7 produced the highest mean accuracy (0.7375), beating kernel size 5 (0.6067) and kernel size 3 (0.5886). For this short training budget, the wider filter seems to capture useful stroke context.

The dropout hypothesis was mixed. No dropout had the best mean accuracy (0.7188), but several of the best individual runs used high dropout. This suggests that dropout interacts strongly with batch size and model capacity in the short-run setting.

The hidden-node hypothesis was only weakly supported. The 50-node hidden layer had the best mean accuracy (0.6530), while 100 and 160 nodes did not improve the average result. Extra dense capacity was not the limiting factor for this experiment.

The batch-size hypothesis was strongly supported for accuracy. Batch size 64 had the best mean accuracy (0.8924), while batch size 512 was much worse (0.3862). Larger batches were often faster per update schedule, but under a fixed epoch count they learned less effectively.

Overall, run 31 had the best accuracy, but run 24 was the better speed/accuracy compromise: it reached 90.55% test accuracy in 4.14 seconds. A final follow-up would retrain the best few architectures for more epochs on the full 60k MNIST training set.
