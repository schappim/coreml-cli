# coreml-cli

A native command-line interface for working with Apple Core ML models on macOS. Inspect, run inference, benchmark, and manage Core ML models without Xcode or Python.

## Features

- **Inspect** - View model structure, inputs/outputs, and metadata
- **Predict** - Run inference on images, text, or JSON data
- **Batch** - Process multiple files with concurrent execution
- **Benchmark** - Measure inference latency and throughput
- **Compile** - Convert `.mlmodel` to optimized `.mlmodelc` format
- **Metadata** - View and manage model metadata

## Installation

### Homebrew (Recommended)

```bash
brew tap schappim/coreml-cli
brew install coreml-cli
```

### Download Binary

Download the latest release from [GitHub Releases](https://github.com/schappim/coreml-cli/releases):

```bash
curl -L https://github.com/schappim/coreml-cli/releases/download/v1.0.0/coreml-1.0.0-macos.tar.gz -o coreml.tar.gz
tar -xzf coreml.tar.gz
sudo mv coreml /usr/local/bin/
```

### Build from Source

Requires macOS 13+ and Swift 5.9+

```bash
git clone https://github.com/schappim/coreml-cli.git
cd coreml-cli
swift build -c release
sudo cp .build/release/coreml /usr/local/bin/
```

### Verify Installation

```bash
coreml --version
# coreml 1.0.0
```

## Usage

### Inspect a Model

View model structure, inputs, outputs, and metadata:

```bash
coreml inspect MobileNetV2.mlmodel
```

Output:
```
Model: MobileNetV2
Size: 24.7 MB
Compiled: No

Inputs:
  image: image 224x224 BGRA32

Outputs:
  classLabel: string
  classLabelProbs: dictionary

Metadata:
  Author: Original Paper: Mark Sandler, Andrew Howard...
  Description: Detects the dominant objects present in an image...
```

JSON output for scripting:
```bash
coreml inspect MobileNetV2.mlmodel --json
```

### Run Inference

Classify an image:

```bash
coreml predict MobileNetV2.mlmodel --input photo.jpg
```

Output:
```
Input: photo.jpg
Inference time: 1.66 ms

Outputs:
  classLabel: golden retriever
  classLabelProbs: golden retriever: 0.8721, Labrador retriever: 0.0543...
```

Save results to file:
```bash
coreml predict MobileNetV2.mlmodel --input photo.jpg --output results.json --json
```

Select compute device:
```bash
coreml predict MobileNetV2.mlmodel --input photo.jpg --device ane  # Apple Neural Engine
coreml predict MobileNetV2.mlmodel --input photo.jpg --device gpu  # GPU
coreml predict MobileNetV2.mlmodel --input photo.jpg --device cpu  # CPU only
```

### Batch Processing

Process a directory of images:

```bash
coreml batch MobileNetV2.mlmodel --dir ./photos --out ./results --format csv
```

Output:
```
Found 100 input files
Results written to: ./results/results.csv

Processed 100 files in 892.45 ms
Average inference time: 2.15 ms
```

Control concurrency:
```bash
coreml batch MobileNetV2.mlmodel --dir ./photos --out ./results --concurrency 8
```

### Benchmark Performance

Measure inference latency:

```bash
coreml benchmark MobileNetV2.mlmodel --input sample.jpg
```

Output:
```
Benchmark Results for: MobileNetV2
==================================================

Configuration:
  Device: all
  Iterations: 100
  Warmup: 10

Latency (ms):
  Mean:      1.279
  Min:       1.008
  Max:       1.602
  StdDev:    0.204

Percentiles (ms):
  P50:       1.200
  P95:       1.523
  P99:       1.589

Throughput: 781.86 inferences/sec
```

Custom iterations:
```bash
coreml benchmark MobileNetV2.mlmodel --input sample.jpg -n 500 --warmup 50
```

JSON output for CI/CD:
```bash
coreml benchmark MobileNetV2.mlmodel --input sample.jpg --json > benchmark.json
```

### Compile Models

Compile `.mlmodel` to optimized `.mlmodelc`:

```bash
coreml compile MobileNetV2.mlmodel
```

Output:
```
Compilation successful!
  Source: /path/to/MobileNetV2.mlmodel
  Output: /path/to/MobileNetV2.mlmodelc
  Original size: 24.7 MB
  Compiled size: 24.5 MB
```

With validation:
```bash
coreml compile MobileNetV2.mlmodel --validate --output-dir ./compiled/
```

### View Metadata

Get model metadata:

```bash
coreml meta get MobileNetV2.mlmodel
```

Output:
```
Metadata for: MobileNetV2.mlmodel

  Author:      Original Paper: Mark Sandler, Andrew Howard...
  Description: Detects the dominant objects present in an image...
  License:     Please see https://github.com/tensorflow/tensorflow...
  Version:     1.0
```

## Command Reference

| Command | Description |
|---------|-------------|
| `coreml inspect <model>` | Inspect model structure and metadata |
| `coreml predict <model> -i <input>` | Run inference on a single input |
| `coreml batch <model> --dir <dir> --out <dir>` | Batch process multiple inputs |
| `coreml benchmark <model> -i <input>` | Benchmark model performance |
| `coreml compile <model>` | Compile model to optimized format |
| `coreml meta get <model>` | View model metadata |

### Global Options

| Option | Description |
|--------|-------------|
| `--json`, `-j` | Output in JSON format |
| `--device <device>` | Compute device: `cpu`, `gpu`, `ane`, or `all` |
| `--help`, `-h` | Show help information |
| `--version` | Show version |

## Supported Input Types

| Type | Extensions | Used For |
|------|------------|----------|
| Images | `.jpg`, `.jpeg`, `.png`, `.heic` | Vision models |
| Audio | `.wav` | Sound classification |
| Text | `.txt` | NLP models |
| Tensors | `.json` | Custom models |

## Examples

### Image Classification Pipeline

```bash
#!/bin/bash
# Classify all images in a folder and generate a report

MODEL="MobileNetV2.mlmodel"
INPUT_DIR="./images"
OUTPUT_DIR="./classifications"

# Run batch classification
coreml batch "$MODEL" --dir "$INPUT_DIR" --out "$OUTPUT_DIR" --format csv

# View results
cat "$OUTPUT_DIR/results.csv"
```

### Performance Comparison

```bash
#!/bin/bash
# Compare inference speed across compute devices

MODEL="MobileNetV2.mlmodel"
INPUT="test.jpg"

echo "CPU Only:"
coreml benchmark "$MODEL" -i "$INPUT" --device cpu -n 50 --json | jq '.meanLatencyMs'

echo "GPU:"
coreml benchmark "$MODEL" -i "$INPUT" --device gpu -n 50 --json | jq '.meanLatencyMs'

echo "Neural Engine:"
coreml benchmark "$MODEL" -i "$INPUT" --device ane -n 50 --json | jq '.meanLatencyMs'
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Benchmark Model
  run: |
    coreml benchmark model.mlmodel -i test.jpg --json > benchmark.json

- name: Check Performance Regression
  run: |
    LATENCY=$(jq '.meanLatencyMs' benchmark.json)
    if (( $(echo "$LATENCY > 10" | bc -l) )); then
      echo "Performance regression detected: ${LATENCY}ms"
      exit 1
    fi
```

### JSON Tensor Input

For models that accept numeric tensor inputs (not images), you can pass JSON arrays:

**Create a JSON input file** (`input.json`):
```json
[5.1, 3.5, 1.4, 0.2]
```

**Run prediction:**
```bash
coreml predict MyClassifier.mlmodel --input input.json
```

**Output:**
```
Input: input.json
Inference time: 0.12 ms

Outputs:
  probabilities: [0.1377, 0.7100, 0.1522]
```

**Batch process multiple JSON files:**
```bash
# Create a directory with JSON input files
mkdir json_samples
echo '[5.1, 3.5, 1.4, 0.2]' > json_samples/sample1.json
echo '[6.7, 3.1, 4.7, 1.5]' > json_samples/sample2.json
echo '[5.9, 3.0, 5.1, 1.8]' > json_samples/sample3.json
echo '[4.6, 3.4, 1.4, 0.3]' > json_samples/sample4.json

# Run batch prediction
coreml batch MyClassifier.mlmodel --dir json_samples --out json_results --format csv
```

**Output CSV** (`json_results/results.csv`):
```csv
input_file,inference_time_ms,probabilities
sample1.json,0.27,"[0.1377, 0.7100, 0.1522]"
sample2.json,0.22,"[0.0613, 0.5931, 0.3456]"
sample3.json,0.29,"[0.0522, 0.5000, 0.4479]"
sample4.json,0.17,"[0.1406, 0.6825, 0.1769]"
```

This is useful for models trained on tabular data, embeddings, or any non-image numeric inputs.

## Requirements

- macOS 13.0 or later
- Apple Silicon or Intel Mac
- Core ML models (`.mlmodel`, `.mlpackage`, or `.mlmodelc`)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Built with [Swift Argument Parser](https://github.com/apple/swift-argument-parser)
- Uses Apple's [Core ML](https://developer.apple.com/documentation/coreml) framework
