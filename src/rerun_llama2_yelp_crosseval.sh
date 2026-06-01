#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o rerun_llama2_yelp_crosseval_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs,gpu-h100,gpu-a100-lowbig
#SBATCH -N 1
#SBATCH -t 06:00:00

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

PROJECT_ROOT=<PROJECT_ROOT>

echo "========================================================="
echo "Cross-eval only: llama2_full_ft yelp row (recovered model)"
echo "Started: $(date)  Node: $(hostname)"
echo "========================================================="

python -c "
import sys, os, torch, logging, pandas as pd
sys.path.insert(0, '${PROJECT_ROOT}/src/Fine_tune/cross_eval')
import llama2_full_ft_cross_eval as ce

logging.basicConfig(level=logging.INFO)
device = ce.EVAL_CONFIG['device']
dtype = ce.EVAL_CONFIG['dtype']

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(ce.BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

csv_path = ce.OUT_CSV
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df.Model_Source != 'yelp']
    all_results = df.to_dict('records')
else:
    all_results = []

yelp_path = os.path.join(ce.FULL_FT_DIR, ce.FT_MODEL_FOLDERS['yelp'])
logging.info(f'Loading yelp model from {yelp_path}')
model = AutoModelForCausalLM.from_pretrained(yelp_path, torch_dtype=dtype, device_map='auto', trust_remote_code=True)
model.eval()

for task_name in ce.VALID_TASKS:
    logging.info(f'  [yelp -> {task_name}]')
    ds = ce.UniversalLlamaDataset(task_name, num_samples=ce.EVAL_CONFIG['eval_num'])
    scores = ce.evaluate_single_run(model, tokenizer, ds, f'yelp->{task_name}')
    logging.info(f'  -> {scores}')
    all_results.append({'Model_Source': 'yelp', 'Eval_Task': task_name, **scores})
    pd.DataFrame(all_results).to_csv(csv_path, index=False)

del model
torch.cuda.empty_cache()
logging.info('Done. CSV updated: ' + csv_path)
"
echo "Cross-eval exit code: $?"

echo ""
echo "========================================================="
echo "Finished: $(date)"
echo "========================================================="
