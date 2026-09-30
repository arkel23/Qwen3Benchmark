"""Write each task's questions.json and summary_keys.json; position is the first offset of the supporting text.

Run after build.py: python make_questions.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# task: ([(question, answer, aliases, supporting text)], summary scope, key facts)
TASKS = {
    'r1_short': ([
        ('How many short-form utterances does one full model sweep evaluate?',
         '159,879', [], 'over 159,879 short-form'),
        ('In Table 2, what is the parameter count in millions of Wh-S?', '241.7', [], '241.7'),
        ('In Table 1, what is the median relative ∆CER (%) at 3 bits for the Low resource tier?', '39.0', [], '39.0'),
        ('In Table 4, which model has the most negative Pearson correlation (rasr) between BFloat16 CER and log '
         'ASR pretraining hours?', 'Voxtral-S', [], 'Voxtral-S'),
        ('Applied to the 4-bit runs, what share of low-tier runs does the five-point absolute degradation threshold '
         'flag as degraded?', '16.8%', ['16.8'], 'the threshold flags 16.8%'),
        ('What Spearman correlation between parameter count and relative penalty is found when restricted to the '
         'Whisper models?', 'ρ=−0.88', ['−0.88'], '(Spearman ρ=−0.88)'),
        ('How many hours of pretraining data does English have?', '438,218 h', ['438,218'], 'English’s 438,218 h'),
        ('Which low-resource-tier language is listed with 42.1 h of pretraining data?',
         'Maltese', [], 'Maltese at 42.1 h'),
        ('What Kruskal–Wallis p-value do the differences between resource tiers obtain across the pooled runs?',
         'p=0.45', ['0.45'], 'Kruskal–Wallis p=0.45'),
        ('What share of runs does the degenerate-run filter remove at 3-bit?', '8.2%', [], 'but 8.2% at 3-bit'),
    ], 'entire context',
        [['quantiz'], ['capacity'], ['resourc', 'pretraining hours', 'low-resource'], ['whisper'],
         ['hqq', 'half-quadratic']]),
    'r2_short': ([
        ('What name do the authors give to their SLAM-ASR models built with Tiny Aya decoders?',
         'LisTAya', [], 'We call the resulting models LisTAya'),
        ('On which GPU is every projector training run performed?', 'NVIDIA H200', ['H200'], 'NVIDIA H200 GPU'),
        ('What learning rate does projector training use with AdamW?',
         '1.5 × 10−3', [], 'learning rate 1.5 × 10−3'),
        ('At 5% significance and 80% power, what is the smallest true advantage the design can reliably detect on '
         'the selection curve?', '4.90 CER', ['4.90'], 'at least 4.90 CER'),
        ('What CER does the best of the three off-the-shelf models reach on Seselwa?',
         '85.18 CER', ['85.18'], 'reaches 85.18 CER'),
        ("In Table 2, what is Tamil's ∆ (matched CER minus the mean of the other two regional variants)?",
         '-14.70', [], '61.21 -14.70'),
        ('How many parameters does the shared Tiny Aya multilingual base model have?', '3.35B', [], '3.35B-parameter'),
        ('In Table 4, what is the loss rise for the Very Low resource tier?',
         '0.211', [], 'falls monotonically from 0.211'),
        ('Which off-the-shelf baseline is the same model as the speech encoder used by the trained systems?',
         'Whisper-Medium', [], 'Whisper-Medium (Radford'),
        ("What share of the Fire variant's post-training mix is English?",
         '46.2%', ['46.2'], '46.2% English against Water'),
    ], 'entire context',
        [['tiny aya'], ['slam-asr'], ['regional', 'region-matched'], ['projector'],
         ['0.55', 'no significant', 'does not']]),
    'r3_medium': ([
        ('In the Mandarin quantization paper, how many evaluation runs does the study comprise?',
         '2,014', [], 'comprising 2,014'),
        ('In the Mandarin quantization paper, to what median CER does Qwen2.5-Omni, the best BFloat16 model, '
         'collapse at HQQ2 on the short-form corpus?', '1575.4', [], 'collapses to 1575.4'),
        ('In the Mandarin quantization paper, what Spearman correlation do CONER and VER have with each other?',
         'ρ = 0.998', ['0.998'], '(ρ = 0.998)'),
        ('In Table 2 of the Mandarin quantization paper, how many utterances does the largest dataset split have?',
         '49,354', [], '49,354'),
        ('In the Mandarin quantization paper, how many utterances does CMMLU-Speech contribute to the long-form '
         'corpus?', '3,147', [], 'contributes 3,147 utterances'),
        ('In the Mandarin quantization paper, what is the diagonal mean of the tone confusion matrix at 2 bits?',
         '57.9%', ['57.9'], 'from 92.2% to 57.9%'),
        ('In the teacher-guided data augmentation (ViTFS) paper, what latency speedup does replacing LayerNorm with '
         'BatchNorm give on the Samsung Galaxy A53 5G at 224 × 224?', '2.64', ['2.64×'], 'by 2.64 times at 224'),
        ('In Table 4 of the ViTFS paper, how many parameters (M) does ViTFS-T have?', '5.63', [], '5.63'),
        ('In the ViTFS paper, what peak CUB top-1 accuracy does the distillation temperature sweep reach?',
         '80.51%', ['80.51'], 'with 80.51%'),
        ('In Table 5 of the ViTFS paper, what is the highest throughput (Imgs/s) among the models compared?',
         '803.0', [], '803.0'),
    ], 'first paper only: Decomposing Quantization Errors in Mandarin Speech Recognition',
        [['quantiz'], ['mandarin'], ['tone'], ['consonant', 'vowel', 'base syllable'], ['capacity', 'larger models']]),
    'r4_medium': ([
        ('In the cultivar identification paper, how many parameters per 200-class task does the warmup method train?',
         '0.51 M', [], '0.51 M parameters per 200-class'),
        ('In the cultivar identification paper, what accuracy does a linear classifier on the frozen representation '
         'reach on SoyGlobal?', '17.9%', ['17.9'], 'reaches only 17.9% on SoyGlobal'),
        ('In the cultivar identification paper, how many images does the pooled warmup corpus contain?',
         '24,367', [], '24,367 images'),
        ('In the cultivar identification paper, over how many cultivar labels does the pooled corpus span?',
         '3,526', [], 'over 3,526 cultivar labels'),
        ('In Table I of the cultivar identification paper, how many classes does SoyGene have?', '1110', [], '1110'),
        ('In Table III of the cultivar identification paper (448 × 448), what is the highest SoyGlobal accuracy of '
         'any method?', '70.69', [], '70.69'),
        ('In the cultivar identification paper, on which GPU is inference throughput measured?',
         'NVIDIA RTX 3090', ['RTX 3090'], 'NVIDIA RTX 3090'),
        ('In the cultivar identification paper, what average accuracy does the previous best PETL method, FFVT, '
         'reach?', '60.9%', ['60.9'], 'average accuracy of 60.9%'),
        ('In the CLCA paper, how many pretrained backbones do the experiments cover?',
         '9 backbones', ['nine backbones'], '9 backbones'),
        ('In the CLCA paper, what is DWG, the number of groups dedicated to each channel in the depth-wise '
         'convolution of the CLA head?', 'DWG = 2', [], 'DWG = 2'),
    ], 'first paper only: Cultivar Identification from Leaf Images with a Semantic Adaptation Warmup for Frozen '
       'Vision Transformers',
        [['cultivar'], ['semantic adaptation warmup', 'warmup'], ['frozen', 'parameter-efficient'], ['5.9', '12.6'],
         ['dual-attention', 'supervised contrastive', 'supcon']]),
    'r5_long': ([
        ('In the ViTFS / teacher-guided data augmentation (MMSys) paper, what CUB top-1 accuracy is obtained at '
         'β = 0.02 in the KD loss weight sweep?', '74.56', [], '74.56'),
        ('How many pretrained checkpoints does the large-scale study on backbone and training and evaluation '
         'settings release?', '825 pretrained checkpoints', ['825'], '825 pretrained checkpoints'),
        ('In the large-scale backbone study, what relative increase in training time does CALMix incur over the '
         'frozen setting?', '1632%', [], '1632% relative increase'),
        ('In the cultivar identification paper, how many images does the pooled warmup corpus contain?',
         '24,367', [], '24,367 images'),
        ('In the Mandarin quantization paper, to what median CER does Qwen2.5-Omni collapse at HQQ2 on the '
         'short-form corpus?', '1575.4', [], 'collapses to 1575.4'),
        ('In the configurable token reduction FPGA accelerator paper, how many frames per second does the design '
         'reach at 224 × 224?', '5291', ['5291 FPS'], 'reaches 5291 frames'),
        ('In the ternary LLM decoder paper, how large is the Bonsai-4B decoder as built?',
         '1,482 MB', ['1,482'], '1,482 MB'),
        ('In the zero-shot spoken language understanding paper, what mean instruction following rate does the '
         'digit template reach?', '81.3%', ['81.3'], 'rises to 81.3%'),
        ('In the zero-shot spoken language understanding paper, how many hours of audio was the Qwen2-Audio-7B '
         'baseline trained on?', '520k hours', ['520k'], '520k hours'),
        ('In the BAQET paper, how many trials does the simulated-annealing search run?',
         '2,000 trials', ['2,000'], '2,000 trials'),
    ], 'first paper only: Application- and Hardware-Aware Backbones for Fine-Grained Image Recognition with '
       'Teacher-Guided Data Augmentation',
        [['teacher-guided data augmentation', 'tgda'], ['vitfs'], ['batchnorm'],
         ['never sees imagenet', 'from scratch', 'random initialisation'],
         ['counterfactual attention learning', 'part attention']]),
    'r6_long': ([
        ('In the regional decoder specialization paper, what name do the authors give to their Tiny Aya based '
         'SLAM-ASR models?', 'LisTAya', [], 'We call the resulting models LisTAya'),
        ('In the regional decoder specialization paper, what CER does the best off-the-shelf model reach on Seselwa?',
         '85.18 CER', ['85.18'], 'reaches 85.18 CER'),
        ('In Table 4 of the regional decoder specialization paper, what is the loss rise for the Very Low resource '
         'tier?', '0.211', [], 'falls monotonically from 0.211'),
        ('In the multilingual ASR quantization paper, how many short-form utterances does one full model sweep '
         'evaluate?', '159,879', [], 'over 159,879 short-form'),
        ('In the SLAM-ASR prompting paper, how long does one projector training run take on one H200 GPU (median)?',
         '15.2 h', [], 'median 15.2 h'),
        ('In the teacher-guided data augmentation letter, up to how many times fewer parameters do LRNets use?',
         '3.9 times', [], 'up to 3.9 times fewer'),
        ('In the large-scale backbone study, what relative increase in training time does CALMix incur over the '
         'frozen setting?', '1632%', [], '1632% relative increase'),
        ('In the ternary LLM decoder paper, how large is the Bonsai-4B decoder as built?',
         '1,482 MB', ['1,482'], '1,482 MB'),
        ('In the configurable token reduction FPGA accelerator paper, what is the latency of the unreduced '
         '288 × 288 baseline?', '430.6 ms', ['430.6'], 'latency falls from 430.6 ms'),
        ('In the configurable token reduction FPGA accelerator paper, by how much does the keep rate 0.7 design '
         'miss timing at 200 MHz?', '57 ps', [], 'misses by 57 ps'),
    ], 'first paper only: How Does Regional Decoder Specialization Help Low-Resource ASR Based on the SLAM-ASR '
       'Framework?',
        [['tiny aya'], ['slam-asr'], ['regional', 'region-matched'], ['projector'],
         ['0.55', 'no significant', 'does not']]),
}


def position(context, support):
    match = re.search(r'\s+'.join(re.escape(part) for part in support.split()), context)
    if match is None:
        raise ValueError(f'supporting text not in context: {support!r}')
    return round(match.start() / len(context), 3)


def main():
    for task, (questions, scope, key_facts) in TASKS.items():
        context = (HERE / task / 'context.txt').read_text()
        rows = [{'id': f'q{i}', 'question': q, 'answer': answer, 'aliases': aliases,
                 'position': position(context, support)}
                for i, (q, answer, aliases, support) in enumerate(questions, 1)]
        (HERE / task / 'questions.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
        summary = {'max_words': 150, 'summary_scope': scope, 'key_facts': key_facts}
        (HERE / task / 'summary_keys.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
        print(task, [row['position'] for row in rows])


if __name__ == '__main__':
    main()
