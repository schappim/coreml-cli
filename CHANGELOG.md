# Changelog

## v1.1.0

### Added

- **`coreml serve`** — run a model as a local HTTP inference API. The model is
  compiled and loaded once at startup and stays warm, so requests skip the
  model-loading cost that a fresh `coreml predict` pays every time.

  ```bash
  coreml serve MobileNetV2.mlmodel
  curl -X POST -H 'Content-Type: image/jpeg' --data-binary @photo.jpg \
    'http://127.0.0.1:8080/v1/predict?top=3'
  ```

  - `GET /health`, `GET /v1/info`, `POST /v1/predict`
  - Input as raw bytes, a `multipart/form-data` upload, or JSON
  - **Multi-input models are reachable for the first time.** `coreml predict`
    feeds one file to the model, so a model with several inputs could not be
    driven from the CLI at all; `{"inputs": {"name": value, …}}` now can.
  - `?top=N` trims classifier dictionaries and adds an ordered `ranked` array,
    since JSON objects carry no ordering
  - Binds loopback by default; `--api-key`, `--cors`, `--allowed-host`,
    `--concurrency`, `--max-body-mb` for anything wider

### Fixed

- **Image prediction was ~6x slower than it needed to be.** A `CIContext` was
  built for every prediction; creating one sets up a GPU pipeline and cost far
  more than the render it was made for. 40 images through `coreml batch` went
  from 2905 ms to 465 ms. This affects `predict` and `batch` as well as `serve`.
- **`coreml meta set --output` could delete the model it was editing.** The
  destination was removed before the source was copied into it, with nothing
  checking the two were different, so `--output` naming the source erased it.
  Such an `--output` is now refused, an existing destination is only replaced
  when it is itself an `.mlpackage`, and the clone is staged and swapped in only
  after the write succeeds.
- **Multi-array outputs were truncated at 100 elements**, so JSON results from
  embedding models silently lost data. The full array is now returned; console
  output still previews the first five.
- **A JSON tensor that did not match the model's input shape** was silently
  truncated or left partly uninitialised, producing garbage predictions. It is
  now an error naming both counts. Non-numeric values in a tensor are rejected
  too, rather than being dropped and shifting every later value out of position.

### Changed

- `swift-argument-parser` 1.7.1 → 1.8.2.
- Dependabot and CI are configured for the repository.

## v1.0.0

Initial release: `inspect`, `predict`, `batch`, `benchmark`, `compile`, `meta`.
