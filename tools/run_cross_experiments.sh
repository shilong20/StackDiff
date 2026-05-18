#!/bin/bash
# =============================================================================
# 跨步数 × 噪声/缺陷率/掺杂率/碳堆积 实验批量运行脚本 (并行版)
# 用途：对不同采样步数测试不同噪声级别、缺陷率、掺杂率或碳堆积强度的性能
# 特性：支持并行任务 (默认3并发)，包含严格的 GPU 状态检查
# 评估任务：shift_pbc
# =============================================================================

set -e  # 遇到错误立即退出

# -----------------------------------------------------------------------------
# 配置区域 - 根据实际情况修改
# -----------------------------------------------------------------------------

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 基础配置文件
BASE_CONFIG="${PROJECT_ROOT}/configs/separate/ReS2.yml"

# 评估配置
BATCH_CONFIG="${PROJECT_ROOT}/generate_sample/ReS2.json"

# 采样步数列表
STEPS=(25 50 100)

# 实验类型: "noise" | "defect" | "dopant" | "carbon" | "all"
EXPERIMENT_TYPE="${1:-noise}"

# 噪声实验：PSNR(vs GT) 级别列表 (dB)
PSNR_LEVELS=(30 26 22 18 15 12)

# 缺陷实验：缺陷率列表
DEFECT_RATES=(0.05 0.10 0.15 0.20)

# 掺杂实验：掺杂率列表（与缺陷率一致）
DOPANT_RATES=(0.05 0.10 0.15 0.20)
# 掺杂实验只跑 step=200
DOPANT_STEPS=(200)

# 碳堆积实验：alpha 强度列表
CARBON_ALPHAS=(0p2 0p4 0p6 0p8)
# 碳堆积实验只跑 step=200
CARBON_STEPS=(200)

# 数据集基础路径
DATA_BASE="${PROJECT_ROOT}/data/experiments"

# GPU 设备 (默认 0)
GPU_ID="${GPU_ID:-0}"

# 并行任务数 (针对 A800 80G，默认 3)
MAX_JOBS=9

# 随机种子
SEED="${SEED:-1234}"

# 日志级别
VERBOSE="${VERBOSE:-info}"

# 错误标志文件 (用于并行进程间通信)
ERROR_FLAG="/tmp/moire_experiment_error_$$"

# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "CRITICAL: nvidia-smi not found! GPU might be unavailable."
        return 1
    fi
    if ! nvidia-smi &> /dev/null; then
        log_error "CRITICAL: nvidia-smi failed! GPU is unresponsive. Please restart the container."
        return 1
    fi
    return 0
}

# 任务包装函数：包含分解和评估
run_task_wrapper() {
    local config_file="$1"
    local seed="$2"
    local verbose="$3"
    local output_dir="$4"
    local out_csv="$5"
    local step="$6"
    local param_desc="$7"

    # 1. 检查全局错误标志
    if [[ -f "$ERROR_FLAG" ]]; then
        return 1
    fi

    # 2. 断点重续：检查输出CSV是否已存在
    if [[ -f "$out_csv" ]]; then
        log_info "Task already completed (found $out_csv), skipping: $param_desc"
        return 0
    fi

    # 3. 任务前再次检查 GPU
    if ! check_gpu; then
        touch "$ERROR_FLAG"
        log_error "GPU check failed before task: $param_desc. Aborting all jobs."
        exit 1
    fi

    log_info "Starting task: $param_desc"

    # 4. 运行分解 (重定向日志以避免控制台混乱)
    log_file="${output_dir}/run.log"
    # 显式传递 CUDA_VISIBLE_DEVICES
    if ! CUDA_VISIBLE_DEVICES="$GPU_ID" python3 "${PROJECT_ROOT}/src/main.py" \
        --config "$config_file" \
        --seed "$seed" \
        --verbose "$verbose" > "$log_file" 2>&1; then
        
        touch "$ERROR_FLAG"
        log_error "Separation failed for task: $param_desc. See $log_file"
        # 显示最后几行错误日志
        tail -n 20 "$log_file" | sed 's/^/[LOG] /'
        exit 1
    fi

    # 5. 运行评估
    eval_log="${output_dir}/eval.log"
    if ! python3 "${PROJECT_ROOT}/tools/evaluate_generate_sample_wraparound_pbc.py" \
        --task shift_pbc \
        --results_root "$output_dir" \
        --batch_config "$BATCH_CONFIG" \
        --out_csv "$out_csv" > "$eval_log" 2>&1; then
        
        touch "$ERROR_FLAG"
        log_error "Evaluation failed for task: $param_desc. See $eval_log"
        tail -n 20 "$eval_log" | sed 's/^/[EVAL] /'
        exit 1
    fi

    log_info "Completed task: $param_desc"
}

# 并行作业控制
wait_for_jobs() {
    local job_limit="$1"
    while [[ $(jobs -r | wc -l) -ge $job_limit ]]; do
        # 检查错误标志
        if [[ -f "$ERROR_FLAG" ]]; then
            log_error "Detected failure in background job. Stopping new jobs."
            wait # 等待剩余作业
            exit 1
        fi
        sleep 1
    done
}

create_temp_config() {
    local base_config="$1"
    local t_sampling="$2"
    local input_dir="$3"
    local output_dir="$4"
    local temp_config="$5"

    cp "$base_config" "$temp_config"

    python3 - "$temp_config" "$t_sampling" "$input_dir" "$output_dir" << 'PYTHON_SCRIPT'
import sys
import yaml

config_path = sys.argv[1]
t_sampling = int(sys.argv[2])
input_dir = sys.argv[3]
output_dir = sys.argv[4]

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

config['sampling']['T_sampling'] = t_sampling
config['paths']['default_input'] = input_dir
config['paths']['default_output'] = output_dir

with open(config_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
PYTHON_SCRIPT
}

# -----------------------------------------------------------------------------
# 噪声实验
# -----------------------------------------------------------------------------

run_noise_experiments() {
    log_info "========== 噪声实验 (Parallel Jobs: $MAX_JOBS) =========="
    log_info "采样步数: ${STEPS[*]}"
    log_info "PSNR 级别: ${PSNR_LEVELS[*]}"

    for step in "${STEPS[@]}"; do
        for psnr in "${PSNR_LEVELS[@]}"; do
            if [[ -f "$ERROR_FLAG" ]]; then return 1; fi

            wait_for_jobs "$MAX_JOBS"

            input_dir="${DATA_BASE}/noise/ReS2_noise_${psnr}_defect0"
            
            if [[ ! -d "$input_dir" ]]; then
                log_error "数据集不存在: $input_dir，跳过"
                continue
            fi

            output_dir="${input_dir}/result_step${step}"
            mkdir -p "$output_dir"

            temp_config="${output_dir}/config_step${step}.yml"
            create_temp_config "$BASE_CONFIG" "$step" "$input_dir" "$output_dir" "$temp_config"

            out_csv="${output_dir}/result_shift_pbc.csv"
            
            # 后台运行
            run_task_wrapper "$temp_config" "$SEED" "$VERBOSE" "$output_dir" "$out_csv" "$step" "step=${step}, PSNR=${psnr}dB" &
        done
    done
    wait
}

# -----------------------------------------------------------------------------
# 缺陷实验
# -----------------------------------------------------------------------------

run_defect_experiments() {
    log_info "========== 缺陷实验 (Parallel Jobs: $MAX_JOBS) =========="
    log_info "采样步数: ${STEPS[*]}"
    log_info "缺陷率: ${DEFECT_RATES[*]}"

    for step in "${STEPS[@]}"; do
        for rate in "${DEFECT_RATES[@]}"; do
            if [[ -f "$ERROR_FLAG" ]]; then return 1; fi

            wait_for_jobs "$MAX_JOBS"

            input_dir="${DATA_BASE}/defect/ReS2_defect${rate}"
            
            if [[ ! -d "$input_dir" ]]; then
                log_error "数据集不存在: $input_dir，跳过"
                continue
            fi

            output_dir="${input_dir}/result_step${step}"
            mkdir -p "$output_dir"

            temp_config="${output_dir}/config_step${step}.yml"
            create_temp_config "$BASE_CONFIG" "$step" "$input_dir" "$output_dir" "$temp_config"

            out_csv="${output_dir}/result_shift_pbc.csv"

            # 后台运行
            run_task_wrapper "$temp_config" "$SEED" "$VERBOSE" "$output_dir" "$out_csv" "$step" "step=${step}, defect_rate=${rate}" &
        done
    done
    wait
}

# -----------------------------------------------------------------------------
# 掺杂实验
# -----------------------------------------------------------------------------

run_dopant_experiments() {
    log_info "========== 掺杂实验 (Parallel Jobs: $MAX_JOBS) =========="
    log_info "采样步数: ${DOPANT_STEPS[*]}"
    log_info "掺杂率: ${DOPANT_RATES[*]}"

    for step in "${DOPANT_STEPS[@]}"; do
        for rate in "${DOPANT_RATES[@]}"; do
            if [[ -f "$ERROR_FLAG" ]]; then return 1; fi

            wait_for_jobs "$MAX_JOBS"

            input_dir="${DATA_BASE}/dopant/ReS2_dopant${rate}"
            
            if [[ ! -d "$input_dir" ]]; then
                log_error "数据集不存在: $input_dir，跳过"
                continue
            fi

            output_dir="${input_dir}/result_step${step}"
            mkdir -p "$output_dir"

            temp_config="${output_dir}/config_step${step}.yml"
            create_temp_config "$BASE_CONFIG" "$step" "$input_dir" "$output_dir" "$temp_config"

            out_csv="${output_dir}/result_shift_pbc.csv"

            run_task_wrapper "$temp_config" "$SEED" "$VERBOSE" "$output_dir" "$out_csv" "$step" "step=${step}, dopant_rate=${rate}" &
        done
    done
    wait
}

# -----------------------------------------------------------------------------
# 碳堆积实验
# -----------------------------------------------------------------------------

run_carbon_experiments() {
    log_info "========== 碳堆积实验 (Parallel Jobs: $MAX_JOBS) =========="
    log_info "采样步数: ${CARBON_STEPS[*]}"
    log_info "Alpha 级别: ${CARBON_ALPHAS[*]}"

    for step in "${CARBON_STEPS[@]}"; do
        for alpha in "${CARBON_ALPHAS[@]}"; do
            if [[ -f "$ERROR_FLAG" ]]; then return 1; fi

            wait_for_jobs "$MAX_JOBS"

            input_dir="${DATA_BASE}/carbon/ReS2_carbon_alpha${alpha}"
            
            if [[ ! -d "$input_dir" ]]; then
                log_error "数据集不存在: $input_dir，跳过"
                continue
            fi

            output_dir="${input_dir}/result_step${step}"
            mkdir -p "$output_dir"

            temp_config="${output_dir}/config_step${step}.yml"
            create_temp_config "$BASE_CONFIG" "$step" "$input_dir" "$output_dir" "$temp_config"

            out_csv="${output_dir}/result_shift_pbc.csv"

            run_task_wrapper "$temp_config" "$SEED" "$VERBOSE" "$output_dir" "$out_csv" "$step" "step=${step}, carbon_alpha=${alpha}" &
        done
    done
    wait
}

# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------

main() {
    rm -f "$ERROR_FLAG"
    trap 'rm -f "$ERROR_FLAG"' EXIT

    log_info "项目根目录: $PROJECT_ROOT"
    log_info "基础配置: $BASE_CONFIG"
    log_info "实验类型: $EXPERIMENT_TYPE"
    log_info "GPU: $GPU_ID (Parallelism: $MAX_JOBS)"

    if ! check_gpu; then
        log_error "Initial GPU check failed. Aborting."
        exit 1
    fi

    if [[ ! -f "$BASE_CONFIG" ]]; then
        log_error "基础配置文件不存在: $BASE_CONFIG"
        exit 1
    fi

    case "$EXPERIMENT_TYPE" in
        noise)
            run_noise_experiments
            ;;
        defect)
            run_defect_experiments
            ;;
        dopant)
            run_dopant_experiments
            ;;
        carbon)
            run_carbon_experiments
            ;;
        all)
            run_noise_experiments
            if [[ ! -f "$ERROR_FLAG" ]]; then
                run_defect_experiments
            fi
            if [[ ! -f "$ERROR_FLAG" ]]; then
                run_dopant_experiments
            fi
            if [[ ! -f "$ERROR_FLAG" ]]; then
                run_carbon_experiments
            fi
            ;;
        *)
            log_error "未知实验类型: $EXPERIMENT_TYPE"
            echo "用法: $0 [noise|defect|dopant|carbon|all]"
            exit 1
            ;;
    esac
    
    if [[ -f "$ERROR_FLAG" ]]; then
        log_error "Experiments completed with ERRORS. Check logs."
        exit 1
    else
        log_info "========== 所有实验完成 =========="
    fi
}

main "$@"
