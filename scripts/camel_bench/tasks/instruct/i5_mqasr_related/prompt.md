Edit the LaTeX paragraph below for a paper. Follow every rule:
1. No sentence longer than 25 words.
2. No semicolons and no em-dashes.
3. Keep every number exactly as written, and add no new numbers.
4. Keep every LaTeX command exactly as written, including citations (\cite, \citep, \citet with their keys), \textbf, \ref and macros such as \NModels{}.
5. Keep the meaning. Do not add or remove claims.
Reply with the edited paragraph only, as plain LaTeX text, without a code block.

Paragraph:
Two recent studies quantize small ASR models. Edge-ASR \citep{feng2025edgeasr} benchmarks eight PTQ methods on Whisper and Moonshine checkpoints up to 244M parameters, and \citet{soehler2025quantizing} compare quantization libraries on Whisper tiny, base and small. Both find that the larger of these small models withstand low bit-widths better, and both evaluate only English. We extend this line in two directions: to billion-scale models, and across \NLangs{} languages whose resourcing spans three orders of magnitude. Our contributions are threefold. We separate the absolute from the relative quantization penalty and show the apparent resourcing effect belongs to the former. We show that model capacity, not resourcing, is what decides how much a model loses. And we show a compressed checkpoint breaking the size trend, losing far more than plain models of its size.
