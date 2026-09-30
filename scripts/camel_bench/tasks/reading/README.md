# Reading tasks

Six tasks ask a model to summarize a paper and answer questions about it. Contexts are plain `pdftotext` extractions of the team's compiled papers. Token counts assume 1 token ≈ 4 characters.

| Task | Sources, in context order | Est. tokens | Questions |
|---|---|---|---|
| `r1_short` | Model Capacity Dominates Language Resourcing in Multilingual ASR Quantization (MultilingualQASR) | 7,246 | 10 |
| `r2_short` | How Does Regional Decoder Specialization Help Low-Resource ASR Based on the SLAM-ASR Framework? (LALMDecoderAnalysis) | 8,482 | 10 |
| `r3_medium` | Decomposing Quantization Errors in Mandarin Speech Recognition (ChineseQASR); Application- and Hardware-Aware Backbones for FGIR with Teacher-Guided Data Augmentation (TGDA_Analysis, mmsys27) | 25,592 | 10 |
| `r4_medium` | Cultivar Identification from Leaf Images with a Semantic Adaptation Warmup (ILA_SAW_Analysis, TAFE26); Cross-Layer Cache Aggregation (ICASSP25_CLCA_Analysis, arXiv) | 29,273 | 10 |
| `r5_long` | TGDA mmsys27; Backbones (Access26); ILA-SAW; ChineseQASR; TR FPGA (DATE27); CLCA; TernaryASR; zero-shot SLU (ZeroShotMultilingualAnalysis, paper_slu); BAQET (TCAS-II brief) | 96,400 | 10 |
| `r6_long` | LALMDecoder; MultilingualQASR; SLAM-ASR prompting (ZeroShotMultilingualAnalysis, paper); BAQET; TGDA letter (spl); Backbones; ILA-SAW; TernaryASR; CLCA; TR FPGA | 91,945 | 10 |

Each task folder holds `context.txt`, `questions.json` and `summary_keys.json`. In concatenated contexts every paper starts with `=== PAPER: <title> ===`, and the summary covers the first paper only. `r6_long` places three answers in the first 10% of the context and two in the last 10%.

Question answers are scored by exact match against the answer and its aliases, after normalizing whitespace and case.
Summaries are scored on staying within `max_words` and on how many key facts they cover; a fact counts when any of its lowercase alternatives appears in the summary.

`build.py` rebuilds the contexts from the PDFs, `make_questions.py` writes the question and summary files, and `validate.py` checks them.
