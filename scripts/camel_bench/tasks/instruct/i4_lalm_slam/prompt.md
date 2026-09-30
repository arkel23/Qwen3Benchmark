Edit the LaTeX paragraph below for a paper. Follow every rule:
1. No sentence longer than 25 words.
2. No semicolons and no em-dashes.
3. Keep every number exactly as written, and add no new numbers.
4. Keep every LaTeX command exactly as written, including citations (\cite, \citep, \citet with their keys), \textbf, \ref and macros such as \NModels{}.
5. Keep the meaning. Do not add or remove claims.
Reply with the edited paragraph only, as plain LaTeX text, without a code block.

Paragraph:
One recent direction addressing these constraints is the SLAM-ASR architecture \citep{maEmbarrassinglySimpleApproach2024}, which connects a pretrained speech encoder to a large language model (LLM) through a trainable linear projector while keeping both pretrained models frozen. Follow-up work has probed the recipe's robustness \citep{kumarPerformanceEvaluationSLAMASR2025}, its data requirements \citep{fongSpeechLLMsLowResource2025}, encoder depth \citep{kolluriRoleEncoderDepth2026}, language-family effects \citep{zhangLanguageFamilyMatters2026}, and prompt sensitivity \citep{burdissoReducingPromptSensitivity2026}. However, the effect of the decoder's \emph{regional specialization} has not been studied. The Tiny Aya family \citep{salamancaTinyAyaBridging2026} makes the missing experiment possible: small multilingual LLMs sharing a common base model and tokenizer, where each regional variant differs only in the post-training data reinforcing its region's languages.
