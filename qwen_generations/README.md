# Qwen generations

`qwen_generation.png` shows MMLU and GSM8K for the ~7-8B base model of Qwen1, Qwen1.5, Qwen2, Qwen2.5 and Qwen3. The GSM8K few-shot setting differs between reports (8-shot for Qwen1, 5-shot for Qwen2, 4-shot from Qwen2.5 on), so the series is a trend, not a controlled comparison.
`qwen_generation_architecture.png` shows GPQA-Diamond in thinking mode for the 27B dense model, the one size released in Qwen3.5, Qwen3.6 and Qwen3.8; Qwen3-32B stands in as the nearest Qwen3 dense model, and Qwen3.7 is absent because it has no 27B release.
The strip under each plot colours every step by what Qwen's own report or model card says changed: Qwen3.5 changes architecture and training data together, while Qwen3.6 and Qwen3.8 keep the Qwen3.5-27B architecture.
Regenerate both figures with `python plot.py` in an environment with pandas, matplotlib and seaborn.
`data.csv` lists, for every plotted value, the source URL, a quote containing the number, the evaluation setting and the Hugging Face repository creation date used as `release`.
