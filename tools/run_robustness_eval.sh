#!/bin/bash
# =============================================================================
# 鲁棒性实验评测脚本：dopant / carbon 分解 + 评估
# 用法：
#   bash tools/run_robustness_eval.sh dopant   # 运行掺杂实验评测
#   bash tools/run_robustness_eval.sh carbon   # 运行碳堆积实验评测
#   bash tools/run_robustness_eval.sh all      # 运行全部
# 可选环境变量：
#   GPU_ID=0          指定 GPU（默认 0）
#   STEP=200          指定采样步数（默认 200）
#   SEED=1234         指定随机种子（默认 1234）
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GPU_ID="${GPU_ID:-0}"
STEP="${STEP:-200}"
SEED="${SEED:-1234}"
BATCH_CONFIG="${PROJECT_ROOT}/generate_sample/batch_config.json"
BASE_CONFIG="${PROJECT_ROOT}/configs/separate/ReS2.yml"
DATA_BASE="${PROJECT_ROOT}/data/experiments"

log_info() { echo "[INFO] $(date '+%H:%M:%S') $*"; }
log_error() { echo "[ERROR] $(date '+%H:%M:%S') $*" >&2; }

run_single() {
    local input_dir="$1"
    local label="$2"

    if [[ ! -d "$input_dir" ]]; then
        log_error "数据集不存在: $input_dir，跳过"
        return 0
    fi

    local output_dir="${input_dir}/result_step${STEP}"
    local config_file="${output_dir}/config_step${STEP}.yml"
    local out_csv="${output_dir}/result_shift_pbc.csv"

    if [[ -f "$out_csv" ]]; then
        log_info "已存在 CSV，跳过: $label"
        return 0
    fi

    mkdir -p "$output_dir"
    cp "$BASE_CONFIG" "$config_file"

    python3 - "$config_file" "$STEP" "$input_dir" "$output_dir" << 'PYSCRIPT'
import sys, yaml
cfg_path, t, inp, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
cfg = yaml.safe_load(open(cfg_path))
cfg['sampling']['T_sampling'] = t
cfg['runtime']['gpu'] = 0
cfg['paths']['default_input'] = inp
cfg['paths']['default_output'] = outp
yaml.safe_dump(cfg, open(cfg_path, 'w'), sort_keys=False, allow_unicode=True)
PYSCRIPT

    log_info "分解: $label (step=$STEP, GPU=$GPU_ID)"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python3 src/main.py \
        --config "$config_file" --seed "$SEED" --verbose info

    log_info "评估: $label"
    python3 tools/evaluate_generate_sample_wraparound_pbc.py \
        --task shift_pbc \
        --results_root "$output_dir" \
        --batch_config "$BATCH_CONFIG" \
        --out_csv "$out_csv"

    log_info "完成: $label -> $out_csv"
}

run_dopant() {
    log_info "========== Dopant 实验 =========="
    for rate in 0.05 0.10 0.15 0.20; do
        run_single "${DATA_BASE}/dopant/ReS2_dopant${rate}" "dopant_rate=${rate}"
    done
}

run_carbon() {
    log_info "========== Carbon 实验 =========="
    for alpha in 0p2 0p4 0p6 0p8; do
        run_single "${DATA_BASE}/carbon/ReS2_carbon_alpha${alpha}" "carbon_alpha=${alpha}"
    done
}

EXPERIMENT_TYPE="${1:-all}"

case "$EXPERIMENT_TYPE" in
    dopant)  run_dopant ;;
    carbon)  run_carbon ;;
    all)     run_dopant; run_carbon ;;
    *)
        log_error "未知实验类型: $EXPERIMENT_TYPE"
        echo "用法: $0 [dopant|carbon|all]"
        exit 1
        ;;
esac

log_info "========== 全部完成 =========="
