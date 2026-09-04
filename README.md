# coreml-cli

A native command-line interface for working with Apple Core ML models on macOS. Inspect, run inference, benchmark, and manage Core ML models without Xcode or Python.

## Features

- **Inspect** - View model structure, inputs/outputs, and metadata
- **Predict** - Run inference on images, text, or JSON data
- **Serve** - Expose a model as a local HTTP inference API
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
curl -L https://github.com/schappim/coreml-cli/releases/download/v1.1.0/coreml-1.1.0-macos.tar.gz -o coreml.tar.gz
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
# 1.1.0
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

### Serve a Model as an HTTP API

Turn any Core ML model into a local REST endpoint. The model is compiled and
loaded once at startup and stays warm, so requests skip the model-loading cost
that a fresh `coreml predict` pays every time.

```bash
coreml serve MobileNetV2.mlmodel
```

```
coreml serve — MobileNetV2

  Listening on http://127.0.0.1:8080
  Device: all · concurrency: 4 · max body: 32 MB

  GET  http://127.0.0.1:8080/health
  GET  http://127.0.0.1:8080/v1/info
  POST http://127.0.0.1:8080/v1/predict

  curl -X POST -H "Content-Type: image/jpeg" --data-binary @photo.jpg http://127.0.0.1:8080/v1/predict

Press Ctrl-C to stop.
```

The banner's `curl` line is generated from the model's own input type, so the
first request is a copy-paste.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Lists the endpoints, input names, and output names |
| `GET` | `/health` | Liveness, uptime, and request/prediction/error counters |
| `GET` | `/v1/info` | Full model description — the JSON `coreml inspect --json` returns |
| `POST` | `/v1/predict` | Run inference |

#### Sending input

Post an image as raw bytes:

```bash
curl -X POST -H "Content-Type: image/jpeg" \
  --data-binary @photo.jpg \
  "http://127.0.0.1:8080/v1/predict?top=3"
```

Or as a file upload:

```bash
curl -X POST -F "file=@photo.jpg" http://127.0.0.1:8080/v1/predict
```

Response:

```json
{
  "model": "MobileNetV2",
  "inferenceTimeMs": 11.6,
  "outputs": {
    "classLabel": "golden retriever",
    "classLabelProbs": { "golden retriever": 0.8721, "Labrador retriever": 0.0543 }
  },
  "ranked": {
    "classLabelProbs": [
      { "label": "golden retriever", "score": 0.8721 },
      { "label": "Labrador retriever", "score": 0.0543 }
    ]
  }
}
```

`?top=N` trims classifier dictionaries to the N highest scores. Because JSON
objects carry no ordering, the same results also come back in `ranked` as an
ordered array.

A tensor model takes a JSON array:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '[5.1, 3.5, 1.4, 0.2]' \
  http://127.0.0.1:8080/v1/predict
```

#### Multi-input models

`coreml predict` feeds one file to the model, so models with several inputs can
only be driven over HTTP. Name each input:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"inputs": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}}' \
  http://127.0.0.1:8080/v1/predict
```

```json
{
  "model": "IrisClassifier",
  "inferenceTimeMs": 0.13,
  "outputs": {
    "species": "setosa",
    "speciesProbability": { "setosa": 0.9458, "versicolor": 0.0542 }
  }
}
```

Values follow each input's declared type: image inputs take a base64 string,
multi-array inputs take a (possibly nested) numeric array, and string, double,
and int64 inputs take their JSON counterparts. A bare `{"input": …}` works when
the model has exactly one input, and an object already keyed by input name is
accepted as-is.

Errors name what went wrong rather than failing silently:

```json
{ "error": { "status": 422, "message": "Missing value for model input 'sepal_width'. Model inputs: petal_length, petal_width, sepal_length, sepal_width" } }
```

#### Serving beyond localhost

`serve` binds `127.0.0.1` by default, so a model is never exposed to the network
by accident. To reach it from elsewhere, bind wider and require a key:

```bash
coreml serve MobileNetV2.mlmodel --host 0.0.0.0 --api-key "$COREML_API_KEY"
```

```bash
curl -X POST -H "X-API-Key: $COREML_API_KEY" \
  -H "Content-Type: image/jpeg" --data-binary @photo.jpg \
  http://192.168.1.10:8080/v1/predict
```

The key is accepted as either `X-API-Key` or `Authorization: Bearer`. Pass
`--cors` to allow calls from a browser page.

On a loopback bind the server only answers requests whose `Host` is a literal IP
address or `localhost`, so a web page that points a domain it controls at
`127.0.0.1` (DNS rebinding) cannot drive your model. If you reach a
loopback-bound server through a name in `/etc/hosts`, list it:

```bash
coreml serve MobileNetV2.mlmodel --allowed-host myapp.local
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port`, `-p` | `8080` | Port to listen on (`0` picks a free one) |
| `--host` | `127.0.0.1` | Interface to bind |
| `--device` | `all` | Compute device: `cpu`, `gpu`, `ane`, or `all` |
| `--concurrency`, `-c` | `4` | Predictions to run at once |
| `--max-body-mb` | `32` | Largest request body accepted |
| `--api-key` | none | Require this key on every request |
| `--cors` | off | Send CORS headers for browser clients |
| `--allowed-host` | none | Extra hostname to accept in the `Host` header (repeatable) |
| `--quiet`, `-q` | off | Do not log requests |

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

### Edit Metadata

Set a metadata field on a `.mlmodel` or `.mlpackage`. The model spec is rewritten
in place at the protobuf level — no Python or `coremltools` required.

```bash
coreml meta set MobileNetV2.mlmodel author "Jane Doe"
coreml meta set MobileNetV2.mlmodel description "Image classifier, ImageNet-1k"
coreml meta set MobileNetV2.mlmodel license "MIT"
coreml meta set MobileNetV2.mlmodel version "1.0.1"
```

Pass an empty string to clear a field:

```bash
coreml meta set MobileNetV2.mlmodel license ""
```

Write the result to a different file instead of overwriting the source:

```bash
coreml meta set MobileNetV2.mlmodel author "Jane" --output MobileNetV2-attributed.mlmodel
```

For `.mlpackage` inputs, `--output` clones the entire package directory and writes
the modified spec inside the clone.

Compiled `.mlmodelc` models are read-only — modify the source `.mlmodel` /
`.mlpackage` and recompile with `coreml compile`.

## Command Reference

| Command | Description |
|---------|-------------|
| `coreml inspect <model>` | Inspect model structure and metadata |
| `coreml predict <model> -i <input>` | Run inference on a single input |
| `coreml serve <model>` | Serve the model as a local HTTP API |
| `coreml batch <model> --dir <dir> --out <dir>` | Batch process multiple inputs |
| `coreml benchmark <model> -i <input>` | Benchmark model performance |
| `coreml compile <model>` | Compile model to optimized format |
| `coreml meta get <model>` | View model metadata |
| `coreml meta set <model> <field> <value>` | Set a metadata field (author, description, license, version) |

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

## Who made this?

CoreML CLI was made by [Marcus Schappi](https://twitter.com/schappim). I create software (and even hardware) for real-world businesses, including:

* **[Little Bird Electronics](https://littlebirdelectronics.com.au/)** — Australia's electronics and STEM store, shipping Australia-wide. We sell [Arduino](https://littlebirdelectronics.com.au/collections/arduino), [Raspberry Pi](https://littlebirdelectronics.com.au/collections/raspberry-pi), [micro:bit](https://littlebirdelectronics.com.au/collections/micro-bit), [STEM and STEAM education kits](https://littlebirdelectronics.com.au/collections/stem-education), [e-textiles](https://littlebirdelectronics.com.au/collections/e-textiles), [robotics](https://littlebirdelectronics.com.au/collections/robotics), [sensors](https://littlebirdelectronics.com.au/collections/sensors) and [electronic components](https://littlebirdelectronics.com.au/collections/components).
* **[Struth.app](https://struth.app/)** — AI runs and grows your trade business. The Struth platform is field service management + CRM + AI.
