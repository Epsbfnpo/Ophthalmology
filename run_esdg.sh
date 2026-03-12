#!/bin/bash

# 建议修改一下输出目录名以区分之前的实验
BASE_OUTPUT_DIR="./output_esdg_multisource_h100"
DOMAINS=("APTOS" "DEEPDR" "FGADR" "IDRID" "MESSIDOR" "RLDR")
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
TIME_LIMIT=36000

echo "========================================================"
echo "🚀 启动 ESDG 批量实验 (多源单目标模式)"
echo "GPU 数量: $NUM_GPUS"
echo "待测试目标域: ${DOMAINS[*]}"
echo "基础输出目录: $BASE_OUTPUT_DIR"
echo "========================================================"

# 【修改点】变量名从 SOURCE 改为了 TARGET，语意更清晰
for TARGET in "${DOMAINS[@]}"
do
    echo ""
    echo "----------------------------------------------------------------"
    echo "▶️  [进度] 正在启动目标域: $TARGET (其余为源域)"
    echo "----------------------------------------------------------------"

    # 【修改点】参数 --source-domain 替换为 --target-domain
    torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 main.py \
        --time-limit $TIME_LIMIT \
        --target-domain $TARGET \
        --output $BASE_OUTPUT_DIR

    if [ $? -ne 0 ]; then
        echo "❌ [错误] 目标域 $TARGET 训练失败！"
    else
        echo "✅ [完成] 目标域 $TARGET 训练结束。"
    fi
    sleep 5
done

echo ""
echo "########################################################"
echo "📊 最终结果汇总 (Running collect_results.py)"
echo "########################################################"
python3 collect_results.py --base_dir "$BASE_OUTPUT_DIR" --domains "${DOMAINS[@]}"

echo "========================================================"
echo "🎉 所有任务执行完毕"
echo "========================================================"
