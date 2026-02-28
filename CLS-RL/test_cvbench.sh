# 把--model_path的路径换成存储的用SAT数据集训练好的权重的路径
# 修改test_cvbench.py中第91行附近的数据集路径
python src/eval/test_cvbench.py --model_path /root/autodl-tmp/CLS-RL/models/Qwen2VL-rl \
    --bs 8 \
    --use_reasoning_prompt