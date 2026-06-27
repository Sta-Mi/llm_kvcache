"""
《高性能计算》课程期末大项目 — Task 1 基础代码
文件名：kv_cache_starter.py

本文件提供了 LLM 自回归推理的 Naive 实现（不使用 KV Cache），
以及完整的实验框架（计时、多轮重复、绘图）。

学生任务：
    在 TODO 标记处，参考 Naive 实现，补全 KV Cache 推理函数
    `generate_with_kv_cache`，并运行 `run_comparison_experiment`
    对比两种实现的性能差异。

运行环境：
    pip install torch transformers matplotlib
    （建议在有 NVIDIA GPU 的环境下运行，CPU 也可运行但速度较慢）
"""

import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────────────────────
# 0. 全局配置
# ─────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2-0.5B"   # 可替换为 "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 等
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT       = "The history of artificial intelligence began"
GEN_LENGTHS  = [10, 50, 100, 200]   # 每次实验生成的 token 数量
REPEAT_TIMES = 3                    # 每个长度重复实验次数，取平均


# ─────────────────────────────────────────────────────────────
# 1. 加载模型与分词器
# ─────────────────────────────────────────────────────────────
def load_model(model_name: str = MODEL_NAME):
    """加载 tokenizer 和模型，返回 (tokenizer, model)。"""
    print(f"[INFO] 正在加载模型：{model_name}  设备：{DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    print("[INFO] 模型加载完成。")
    return tokenizer, model


# ─────────────────────────────────────────────────────────────
# 2. Naive 推理（已实现，禁止修改）
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_naive(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
) -> tuple[list[float], str]:
    """
    Naive 自回归推理：每步将全部历史 token 重新输入模型。

    返回：
        per_token_latencies : 每生成一个 token 的耗时列表（秒）
        generated_text      : 最终生成的完整文本
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    per_token_latencies = []

    for _ in range(max_new_tokens):
        t0 = time.perf_counter()
        # ── 每次将全部历史 token 重新输入模型 ──
        outputs = model(input_ids=input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        t1 = time.perf_counter()
        per_token_latencies.append(t1 - t0)

        # 遇到 EOS 提前结束
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return per_token_latencies, generated_text


# ─────────────────────────────────────────────────────────────
# 3. KV Cache 推理（学生实现)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_with_kv_cache(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
) -> tuple[list[float], str]:
    """
    带 KV Cache 的自回归推理：利用模型返回的 past_key_values，
    每步仅输入最新生成的单个 token。

    返回：
        per_token_latencies : 每生成一个 token 的耗时列表（秒）
        generated_text      : 最终生成的完整文本

    提示：
        - 第一步（Prefill）：将完整 prompt 输入模型，获取 past_key_values。
        - 后续步骤（Decode）：仅输入上一步生成的 next_token_id，
          并将 past_key_values 传入 model()，模型会自动更新并返回新的
          past_key_values。
        - 参考 Hugging Face 文档中 `use_cache=True` 和
          `past_key_values` 参数的说明。
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    per_token_latencies = []

    # ── TODO：在此处实现 KV Cache 推理逻辑 ──────────────────
    # 步骤提示：
    #   1. 使用完整 input_ids 进行第一次前向传播（Prefill），
    #      令 use_cache=True，保存返回的 past_key_values。
    #   2. 取出 logits[:, -1, :] 得到第一个 next_token，计时。
    #   3. 循环 max_new_tokens - 1 次：
    #      a. 仅将 next_token_id（形状 [1,1]）作为 input_ids 输入，
    #         同时传入 past_key_values。
    #      b. 更新 past_key_values 为新返回的值。
    #      c. 计时并记录延迟。
    #      d. 遇到 EOS 提前退出。
    #   4. 拼接所有生成的 token，解码为文本。

    raise NotImplementedError("请在此处实现 KV Cache 推理逻辑。")
    # ── TODO 结束 ────────────────────────────────────────────

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return per_token_latencies, generated_text


# ─────────────────────────────────────────────────────────────
# 4. 实验框架：多轮计时 + 汇总
# ─────────────────────────────────────────────────────────────
def measure_latency(
    generate_fn,
    tokenizer,
    model,
    prompt: str,
    gen_length: int,
    repeat: int = REPEAT_TIMES,
) -> dict:
    """
    对 generate_fn 重复 repeat 次，取平均延迟。

    返回字典：
        total_latency_avg   : 平均总延迟（秒）
        per_token_avg       : 平均每 token 延迟（秒）
        per_token_series    : 最后一次实验的逐 token 延迟列表
    """
    total_latencies = []
    last_series = []

    for i in range(repeat):
        latencies, _ = generate_fn(tokenizer, model, prompt, gen_length)
        total_latencies.append(sum(latencies))
        last_series = latencies

    total_avg    = sum(total_latencies) / len(total_latencies)
    per_tok_avg  = total_avg / gen_length

    return {
        "total_latency_avg": total_avg,
        "per_token_avg":     per_tok_avg,
        "per_token_series":  last_series,
    }


# ─────────────────────────────────────────────────────────────
# 5. 对比实验主函数
# ─────────────────────────────────────────────────────────────
def run_comparison_experiment(tokenizer, model):
    """
    在 GEN_LENGTHS 定义的多个生成长度下，分别测量 Naive 和 KV Cache
    两种实现的推理延迟，并绘制对比曲线图。
    """
    results_naive = []
    results_kvcache = []

    for length in GEN_LENGTHS:
        print(f"\n[实验] 生成长度 = {length} token ...")

        print("  → Naive 推理 ...")
        r_naive = measure_latency(generate_naive, tokenizer, model, PROMPT, length)
        results_naive.append(r_naive)
        print(f"     总延迟（均值）: {r_naive['total_latency_avg']:.3f}s  "
              f"每 token（均值）: {r_naive['per_token_avg']*1000:.2f}ms")

        print("  → KV Cache 推理 ...")
        r_kv = measure_latency(generate_with_kv_cache, tokenizer, model, PROMPT, length)
        results_kvcache.append(r_kv)
        print(f"     总延迟（均值）: {r_kv['total_latency_avg']:.3f}s  "
              f"每 token（均值）: {r_kv['per_token_avg']*1000:.2f}ms")

    # ── 绘图 ──────────────────────────────────────────────────
    total_naive   = [r["total_latency_avg"]  for r in results_naive]
    total_kv      = [r["total_latency_avg"]  for r in results_kvcache]
    pertok_naive  = [r["per_token_avg"]*1000 for r in results_naive]
    pertok_kv     = [r["per_token_avg"]*1000 for r in results_kvcache]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Naive vs. KV Cache 推理性能对比", fontsize=14)

    # 图1：总延迟
    axes[0].plot(GEN_LENGTHS, total_naive, marker="o", label="Naive")
    axes[0].plot(GEN_LENGTHS, total_kv,    marker="s", label="KV Cache")
    axes[0].set_xlabel("生成 Token 数量")
    axes[0].set_ylabel("总推理延迟 (s)")
    axes[0].set_title("总推理延迟对比")
    axes[0].legend()
    axes[0].grid(True)

    # 图2：平均每 token 延迟
    axes[1].plot(GEN_LENGTHS, pertok_naive, marker="o", label="Naive")
    axes[1].plot(GEN_LENGTHS, pertok_kv,    marker="s", label="KV Cache")
    axes[1].set_xlabel("生成 Token 数量")
    axes[1].set_ylabel("平均每 Token 延迟 (ms)")
    axes[1].set_title("平均每 Token 延迟对比")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("kv_cache_comparison.png", dpi=150)
    print("\n[INFO] 对比图已保存为 kv_cache_comparison.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 6. 入口
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tokenizer, model = load_model()
    run_comparison_experiment(tokenizer, model)
