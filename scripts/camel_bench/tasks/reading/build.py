"""Build the reading tasks' context.txt from the team's compiled paper PDFs.

Run on server-3090 (needs pdftotext and /mnt/ssd2tb/edwin/analysis): python build.py [--stats | --dump KEY]
"""
import argparse
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = Path('/mnt/ssd2tb/edwin/analysis')

# key: (title, pdf relative to ANALYSIS)
PAPERS = {
    'mqasr': ('Model Capacity Dominates Language Resourcing in Multilingual ASR Quantization',
              'MultilingualQASR/paper/main.pdf'),
    'lalm': ('How Does Regional Decoder Specialization Help Low-Resource ASR Based on the SLAM-ASR Framework?',
             'LALMDecoderAnalysis/paper/main.pdf'),
    'cqasr': ('Decomposing Quantization Errors in Mandarin Speech Recognition', 'ChineseQASR/paper/main_ojsp.pdf'),
    'ternary': ('Ternary LLM Decoders for Multilingual Speech Recognition with a Frozen Encoder and a Linear Projector',
                'TernaryASRAnalysis/paper/main.pdf'),
    'tgda_mm': ('Application- and Hardware-Aware Backbones for Fine-Grained Image Recognition with Teacher-Guided '
                'Data Augmentation', 'TGDA_Analysis/paper/mmsys27/main.pdf'),
    'tgda_spl': ('Enabling Fine-Grained Image Recognition from Scratch with Teacher-Guided Data Augmentation',
                 'TGDA_Analysis/paper/spl/main.pdf'),
    'zs': ('How to Prompt Monolingual SLAM-ASR Models Towards Zero-Shot Multilingual Speech Transcription and '
           'Translation', 'ZeroShotMultilingualAnalysis/paper/main.pdf'),
    'zs_slu': ('Zero-Shot Spoken Language Understanding with SLAM-ASR: What Do Monolingual ASR-Trained Linear '
               'Projectors Enable?', 'ZeroShotMultilingualAnalysis/paper_slu/main.pdf'),
    'ila': ('Cultivar Identification from Leaf Images with a Semantic Adaptation Warmup for Frozen Vision Transformers',
            'ILA_SAW_Analysis/drafts/TAFE26/main.pdf'),
    'backbones': ('A Large-Scale Study on the Interaction between Backbone and Training and Evaluation Setting in '
                  'Fine-Grained Image Recognition', 'BackbonesAnalysis/drafts/Access26_Backbones/main.pdf'),
    'trfpga': ('Configurable Token Reduction in a Stream-Based Vision Transformer Accelerator on FPGA',
               'TR_FPGA_Analysis/submissions/DATE27_TR_FPGA/main.pdf'),
    'baqet': ('BAQET: BRAM-Aware Quantization for Efficient Transformer Inference via a Stream-Based Architecture on '
              'an FPGA', 'TR_FPGA_Analysis/submissions/TCASII_BAQET_brief/main.pdf'),
    'clca': ('Cross-Layer Cache Aggregation for Token Reduction in Ultra-Fine-Grained Image Recognition',
             'ICASSP25_CLCA_Analysis/icassp25_clca_arxiv.pdf'),
}

# task: paper keys in context order; the summary request covers the first paper
TASKS = {
    'r1_short': ['mqasr'],
    'r2_short': ['lalm'],
    'r3_medium': ['cqasr', 'tgda_mm'],
    'r4_medium': ['ila', 'clca'],
    'r5_long': ['tgda_mm', 'backbones', 'ila', 'cqasr', 'trfpga', 'clca', 'ternary', 'zs_slu', 'baqet'],
    'r6_long': ['lalm', 'mqasr', 'zs', 'baqet', 'tgda_spl', 'backbones', 'ila', 'ternary', 'clca', 'trfpga'],
}

EMAIL = re.compile(r'\S+@\S+\.\S+')
FOOTERS = re.compile(r'(Submitted to .*Do not distribute\.|Confidential Review Copy\. DO NOT DISTRIBUTE\.'
                     r'|<Society logo\(s\) and publication title will appear here\.>)')


def drop_line_numbers(lines):
    """Remove runs (>=4, blank lines allowed between) of standalone consecutive integers: review line numbering."""
    keep = [True] * len(lines)
    i = 0
    while i < len(lines):
        if not re.fullmatch(r'\d{1,4}', lines[i].strip()):
            i += 1
            continue
        run, prev, j = [i], int(lines[i]), i + 1
        while j < len(lines):
            s = lines[j].strip()
            if s == '':
                j += 1
                continue
            if re.fullmatch(r'\d{1,4}', s) and int(s) == prev + 1:
                run.append(j)
                prev += 1
                j += 1
                continue
            break
        if len(run) >= 4:
            for k in run:
                keep[k] = False
            i = j
        else:
            i += 1
    return [line for line, k in zip(lines, keep) if k]


def extract(key):
    pdf = ANALYSIS / PAPERS[key][1]
    raw = subprocess.run(['pdftotext', str(pdf), '-'], capture_output=True, text=True, check=True).stdout
    lines = [line for line in raw.replace('\f', '\n').split('\n')
             if not EMAIL.search(line) and not FOOTERS.search(line)]
    text = '\n'.join(drop_line_numbers(lines))
    return re.sub(r'\n{3,}', '\n\n', text).strip() + '\n'


def build(task):
    keys = TASKS[task]
    if len(keys) == 1:
        return extract(keys[0])
    return '\n'.join(f'=== PAPER: {PAPERS[k][0]} ===\n\n{extract(k)}' for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--dump')
    args = ap.parse_args()
    if args.dump:
        print(extract(args.dump))
        return
    if args.stats:
        for key in PAPERS:
            print(f'{key:10s} {len(extract(key)):7d} chars')
        return
    for task in TASKS:
        text = build(task)
        (HERE / task).mkdir(exist_ok=True)
        (HERE / task / 'context.txt').write_text(text)
        print(f'{task}: {len(text)} chars, ~{len(text) // 4} tokens')


if __name__ == '__main__':
    main()
