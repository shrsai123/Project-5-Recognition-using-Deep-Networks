# Project 5: Recognition Using Deep Networks

This folder contains a PyTorch implementation of a convolutional neural network for MNIST digit recognition.

## Submission Note

I am using 1 time travel day for this project.

## Run

```powershell
py -3.11 mnist_cnn.py
```

The default run trains for 5 epochs, saves the trained network to `mnist_cnn.pt`, and writes report figures to `outputs/`:

- `first_six_test_digits.png`
- `network_diagram.png`
- `training_testing_error.png`
- `training_testing_accuracy.png`

Useful options:

```powershell
py -3.11 mnist_cnn.py --epochs 5 --batch-size 64 --lr 0.01
```

## Run The Saved Network

```powershell
py -3.11 run_saved_network.py
```

This reads `mnist_cnn.pt`, sets the model to evaluation mode, prints the first 10 test-set output vectors, and saves:

- `outputs/test_set_predictions.txt`
- `outputs/plot_predictions.png`

## Custom Handwritten Digits

Put cropped digit images in a folder, then run:

```powershell
py -3.11 classify_handwritten_digits.py path\to\digit_images
```

The script assumes black marker on white paper and inverts the image to match MNIST's white digits on a black background. If your images are already white on black, use:

```powershell
py -3.11 classify_handwritten_digits.py path\to\digit_images --no-invert
```

The custom digit script saves:

- `outputs/custom_digit_predictions.txt`
- `outputs/custom_digit_predictions.png`

For one photo containing multiple handwritten digits, run:

```powershell
py -3.11 classify_digit_sheet.py handwritten_digits.jpeg --labels 1,2,4,6,8,9,0,3,5,7
```

This auto-crops the sheet and saves results to `outputs/custom_sheet/`.

## Examine The Network

```powershell
py -3.11 examine_network.py
```

This reads `mnist_cnn.pt`, prints the model and `conv1` filter weights, and saves:

- `outputs/network_analysis.txt`
- `outputs/conv1_filters.png`
- `outputs/conv1_filter_effects.png`

## Examine A Pretrained Network

```powershell
py -3.11 examine_pretrained_resnet.py
```

This loads pretrained torchvision ResNet-18 and saves:

- `outputs/pretrained_resnet/pretrained_resnet_analysis.txt`
- `outputs/pretrained_resnet/resnet_conv1_filters.png`
- `outputs/pretrained_resnet/resnet_layer1_0_conv1_filters.png`
- `outputs/pretrained_resnet/resnet_conv1_filter_effects.png`
- `outputs/pretrained_resnet/resnet_layer1_0_conv1_filter_effects.png`

## Greek Transfer Learning

Extract `greek_train.zip` so the project has `greek_train\alpha`, `greek_train\beta`, and `greek_train\gamma`, then run:

```powershell
py -3.11 transfer_greek.py
```

This reads the MNIST model, freezes its existing weights, replaces `fc2` with a 3-output layer, and saves:

- `greek_cnn.pt`
- `outputs/greek_transfer_log.txt`
- `outputs/greek_training_accuracy.png`
- `outputs/greek_predictions.png`

To test your own cropped Greek-letter images:

```powershell
py -3.11 transfer_greek.py --custom-path path\to\custom_greek_images
```

For one photo containing rows of handwritten alpha, beta, and gamma symbols, first crop it into the expected folder layout:

```powershell
py -3.11 prepare_greek_sheet.py handwritten_greek_letters.jpeg
py -3.11 transfer_greek.py --custom-path custom_greek_images
```

For automatic correctness reporting, organize the custom images like this:

```text
custom_greek_images\
  alpha\
    alpha_01.jpg
  beta\
    beta_01.jpg
  gamma\
    gamma_01.jpg
```

The script writes `outputs/custom_greek_predictions.txt` and `outputs/custom_greek_predictions.png`.

Additional handwritten Greek examples for submission: [Google Drive custom Greek images](https://drive.google.com/file/d/1mYxKgZVV8ah8fwp9LznAUQjjSWv6sb9N/view?usp=sharing).

## Greek Transfer Extension

```powershell
py -3.11 experiment_greek_transfer.py
```

This evaluates transfer-learning dimensions for Task 3 and saves:

- `outputs/greek_experiment/greek_transfer_experiment_results.csv`
- `outputs/greek_experiment/greek_transfer_experiment_results.json`
- `outputs/greek_experiment/greek_transfer_experiment_report.md`
- `outputs/greek_experiment/epochs_to_target.png`
- `outputs/greek_experiment/dimension_success.png`

## Transformer MNIST Network

```powershell
py -3.11 mnist_transformer.py
```

This replaces the convolution layers with patch tokens and transformer encoder layers, then saves:

- `mnist_transformer.pt`
- `outputs/transformer_training_log.txt`
- `outputs/transformer_network_diagram.png`
- `outputs/transformer_training_testing_error.png`
- `outputs/transformer_training_testing_accuracy.png`

## Fixed Gabor First Layer

```powershell
py -3.11 mnist_gabor_fixed.py
```

This replaces the first MNIST conv layer with a frozen Gabor filter bank and saves:

- `mnist_gabor_fixed.pt`
- `outputs/gabor_fixed/gabor_training_log.txt`
- `outputs/gabor_fixed/gabor_filter_bank.png`
- `outputs/gabor_fixed/gabor_training_testing_accuracy.png`
- `outputs/gabor_fixed/gabor_training_testing_error.png`

## Advanced Transformer

```powershell
py -3.11 mnist_transformer_advanced.py
```

This uses overlapping convolutional patch embedding and a fused CLS-plus-mean classifier, then saves:

- `mnist_transformer_advanced.pt`
- `outputs/advanced_transformer/advanced_transformer_training_log.txt`
- `outputs/advanced_transformer/advanced_transformer_diagram.png`
- `outputs/advanced_transformer/advanced_transformer_training_testing_accuracy.png`
- `outputs/advanced_transformer/advanced_transformer_training_testing_error.png`

## Automated CNN Experiment

```powershell
py -3.11 experiment_cnn.py
```

This evaluates 50 CNN variations and saves:

- `outputs/experiment/cnn_experiment_results.csv`
- `outputs/experiment/cnn_experiment_results.json`
- `outputs/experiment/experiment_report.md`
- `outputs/experiment/accuracy_vs_time.png`
- `outputs/experiment/dimension_effects.png`

## Network

The `MyNetwork` class uses the required architecture:

- 10 filters of size 5x5
- 2x2 max pooling with ReLU
- 20 filters of size 5x5
- 50% dropout
- 2x2 max pooling with ReLU
- flattening
- fully connected layer with 50 nodes and ReLU
- final fully connected layer with 10 nodes and `log_softmax`
