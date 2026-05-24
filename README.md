# gemma4-bench

Benchmark runs for **Gemma 4 26B-A4B-it** quantizations on long-context evaluation suites, on a single NVIDIA RTX Pro 6000 Workstation (96 GB).

## 📊 Dashboard

**[https://codeandcodes.github.io/gemma4-bench/dashboard.html](https://codeandcodes.github.io/gemma4-bench/dashboard.html)** — live cross-quant comparison tables with per-task, per-length, per-difficulty breakdowns.

Also available locally as [`dashboard.html`](./dashboard.html) (self-contained, opens in any browser) and [`dashboard.csv`](./dashboard.csv) (long-format, for spreadsheets).

## Benchmarks

| Benchmark                                                              | Folder                              | Status                                                                                  |
| ---------------------------------------------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------- |
| [LongBench Pro](https://huggingface.co/datasets/caskcsg/LongBench-Pro) | [`longbench-pro/`](./longbench-pro) | Unsloth Q8/Q4/IQ2 + Bartowski Q8/IQ4/IQ2 done on 500-item subset; full runs in progress |

## Hardware / Software

- **GPU:** NVIDIA RTX Pro 6000 Workstation Edition (96 GB GDDR7)
- **Server:** [llama.cpp](https://github.com/ggml-org/llama.cpp) via [llama-swap](https://github.com/mostlygeek/llama-swap)
- **Models:** [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) and [bartowski/google_gemma-4-26B-A4B-it-GGUF](https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF)

## For downstream tools / agents

See [`longbench-pro/SCHEMA.md`](./longbench-pro/SCHEMA.md) for the JSON schema of every result file, plus Python snippets for common queries.

## License

Code: MIT. Results: shared under the same terms as the underlying benchmark dataset.
