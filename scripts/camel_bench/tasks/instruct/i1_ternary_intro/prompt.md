Edit the LaTeX paragraph below for a paper. Follow every rule:
1. No sentence longer than 25 words.
2. No semicolons and no em-dashes.
3. Keep every number exactly as written, and add no new numbers.
4. Keep every LaTeX command exactly as written, including citations (\cite, \citep, \citet with their keys), \textbf, \ref and macros such as \NModels{}.
5. Keep the meaning. Do not add or remove claims.
Reply with the edited paragraph only, as plain LaTeX text, without a code block.

Paragraph:
A speech large language model (LLM) for recognition is usually three parts: a speech encoder, a text LLM decoder, and a projector that maps encoder frames into the decoder's embedding space \cite{ma_embarrassingly_2024,chu2024qwen2audio}. The decoder holds most of the parameters and most of the memory traffic at decode time, so it is the part worth compressing. Ternary LLMs store every linear weight as one of three values \cite{ma2024bitnet158,wang2023bitnet}, and open ternary decoders now exist at useful sizes: Falcon-E trained with quantization-aware training \cite{falconllm2025falcone}, and Ternary-Bonsai distilled from Qwen3 \cite{prismml2026bonsaiternary,yang_qwen3_2025}. No published result places one of them behind a speech encoder.
