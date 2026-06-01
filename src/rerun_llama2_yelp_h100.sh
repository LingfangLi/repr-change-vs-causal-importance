#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o rerun_llama2_yelp_h100_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-h100,gpu-a100-lowbig
#SBATCH -N 1
#SBATCH -t 1-00:00:00

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export CUDA_LAUNCH_BLOCKING=1

PROJECT_ROOT=<PROJECT_ROOT>
MODEL_DIR=<DATA_ROOT>/fine_tuned_model

echo "========================================================="
echo "Re-running all llama2_full_ft yelp experiments with recovered yelp model"
echo "Started: $(date)  Node: $(hostname)"
echo "========================================================="

# -------------------------------------------------------
# 1. EAP edges (yelp finetuned)
# -------------------------------------------------------
echo ""
echo ">>> [1/4] EAP edges: llama2 yelp finetuned"
python ${PROJECT_ROOT}/src/EAP/eap_unified.py \
    --task yelp \
    --model_name llama2 \
    --base_model_name meta-llama/Llama-2-7b-hf \
    --ft_model_path ${MODEL_DIR}/llama2-7b-yelp-full \
    --data_path ${PROJECT_ROOT}/output/corrupted_data/yelp_corrupted.csv \
    --output_dir ${PROJECT_ROOT}/output/EAP_edges/finetuned/ \
    --mode finetuned \
    --top_k 400
echo "EAP exit code: $?"

# -------------------------------------------------------
# 2. Attention KL (sentiment_yelp)
# -------------------------------------------------------
echo ""
echo ">>> [2/4] Attention KL: llama2_full_ft sentiment_yelp"
python -c "
import sys, os, logging
sys.path.insert(0, os.path.join('${PROJECT_ROOT}', 'experiments', 'attention_matrix_analysis'))
import measure_attention_kl as m
m.UserConfig.RUN_MODE = 'SINGLE'
m.UserConfig.TARGET_MODEL = 'llama2_full_ft'
m.UserConfig.TARGET_TASK = ['sentiment_yelp']
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
for task in m.UserConfig.TARGET_TASK:
    m.AnalysisEngine.run_pair(m.UserConfig.TARGET_MODEL, task)
"
echo "Attention KL exit code: $?"

# -------------------------------------------------------
# 3. Induction head (sentiment_yelp)
# -------------------------------------------------------
echo ""
echo ">>> [3/4] Induction head: llama2_full_ft sentiment_yelp"
TASKS=sentiment_yelp python ${PROJECT_ROOT}/experiments/induction_head/detect_induction_head_llama2_full.py
echo "Induction head exit code: $?"

# -------------------------------------------------------
# 4. Cross-eval (yelp row only — 6 eval tasks)
# -------------------------------------------------------
echo ""
echo ">>> [4/4] Cross-eval: llama2_full_ft yelp row"
python -c "
import sys, os, torch, logging, pandas as pd
sys.path.insert(0, '${PROJECT_ROOT}/src/Fine_tune/cross_eval')
import llama2_full_ft_cross_eval as ce

logging.basicConfig(level=logging.INFO)
device = ce.EVAL_CONFIG['device']
dtype = ce.EVAL_CONFIG['dtype']

# Load tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(ce.BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load existing CSV, remove old yelp rows
csv_path = ce.OUT_CSV
df = pd.read_csv(csv_path)
df = df[df.Model_Source != 'yelp']
all_results = df.to_dict('records')

# Load yelp FT model
yelp_path = os.path.join(ce.FULL_FT_DIR, ce.FT_MODEL_FOLDERS['yelp'])
logging.info(f'Loading yelp model from {yelp_path}')
model = AutoModelForCausalLM.from_pretrained(yelp_path, torch_dtype=dtype, device_map='auto', trust_remote_code=True)
model.eval()

for task_name in ce.VALID_TASKS:
    logging.info(f'  [yelp → {task_name}]')
    ds = ce.UniversalDataset(task_name, num_samples=ce.EVAL_CONFIG['eval_num'])
    scores = ce.evaluate_single_run(model, tokenizer, ds, f'yelp->{task_name}', device)
    logging.info(f'  -> {scores}')
    all_results.append({'Model_Source': 'yelp', 'Eval_Task': task_name, **scores})
    pd.DataFrame(all_results).to_csv(csv_path, index=False)

del model
torch.cuda.empty_cache()
logging.info('Done. CSV updated.')
"
echo "Cross-eval exit code: $?"

echo ""
echo "========================================================="
echo "All done: $(date)"
echo "========================================================="
