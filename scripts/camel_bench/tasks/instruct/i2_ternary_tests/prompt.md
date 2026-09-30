Edit the LaTeX paragraph below for a paper. Follow every rule:
1. No sentence longer than 25 words.
2. No semicolons and no em-dashes.
3. Keep every number exactly as written, and add no new numbers.
4. Keep every LaTeX command exactly as written, including citations (\cite, \citep, \citet with their keys), \textbf, \ref and macros such as \NModels{}.
5. Keep the meaning. Do not add or remove claims.
Reply with the edited paragraph only, as plain LaTeX text, without a code block.

Paragraph:
\textbf{Populations and tests.} Three populations are compared: the selection split, where the best checkpoint is chosen; the FLEURS test set \cite{conneau2022fleurs}, available for \NumFleursLocales{} of the locales; and the in-domain WorldSpeech test set. Each ternary arm is paired with the FP16 arm per locale. We report the mean and median paired difference, a bootstrap 95\% confidence interval on the mean, a two-sided Wilcoxon signed-rank $p$ \cite{wilcoxon_individual_1945}, a Holm correction across the four arms \cite{holm_simple_1979}, and the minimum detectable effect (MDE) at 80\% power. A positive difference means the ternary arm is worse.
