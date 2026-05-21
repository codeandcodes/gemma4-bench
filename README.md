# gemma4-bench

Benchmark runs for **Gemma 4 26B-A4B-it** quantizations on long-context evaluation suites, on a single NVIDIA RTX Pro 6000 Workstation (96 GB).

## Benchmarks

| Benchmark                                                              | Folder                              | Status                                     |
| ---------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------ |
| [LongBench Pro](https://huggingface.co/datasets/caskcsg/LongBench-Pro) | [`longbench-pro/`](./longbench-pro) | Q8_K_XL complete; Q4_K_M and IQ2_M running |

## Hardware / Software

- **GPU:** NVIDIA RTX Pro 6000 Workstation Edition (96 GB GDDR7)
- **Server:** [llama.cpp](https://github.com/ggml-org/llama.cpp) via [llama-swap](https://github.com/mostlygeek/llama-swap)
- **Model:** [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) (various quants)

## License

Code: MIT. Results: shared under the same terms as the underlying benchmark dataset.
