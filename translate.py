import torch
from transformer_first import Transformer
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# === 加载模型与 Tokenizer ===
checkpoint_file = "/root/xyy/checkpoint_epoch_8.pth"
model_path = "/root/xyy/token_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer 配置      
d_model = 512
d_ff = 2048
d_k = d_v = 64
n_heads = 8
n_layers = 6
drop_prob = 0.1
max_length = 128
batch_size = 16

# 加载 Tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = Transformer(
    src_pad_idx=tokenizer.pad_token_id,
    trg_pad_idx=tokenizer.pad_token_id,
    enc_voc_size=tokenizer.vocab_size,
    dec_voc_size=tokenizer.vocab_size,
    d_model=d_model,
    max_len=max_length,
    n_head=n_heads,
    ffn_hidden=d_ff,
    n_layer=n_layers,
    drop_prob=drop_prob,
    device=device,
).to(device)

# 加载模型检查点
checkpoint = torch.load(checkpoint_file, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# === 翻译函数，返回 token 列表 ===
def translate_to_tokens(sentence, tokenizer, model, device, max_len=128):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        src = inputs["input_ids"].to(device)

        trg_indices = [tokenizer.cls_token_id]
        for _ in range(max_len):
            trg_tensor = torch.tensor([trg_indices], dtype=torch.long).to(device)
            output = model(src, trg_tensor)
            next_token = output[0, -1, :].argmax(dim=-1).item()
            trg_indices.append(next_token)
            if next_token == tokenizer.sep_token_id:
                break

        # 返回 tokens（去除 BOS 和 EOS）
        return tokenizer.convert_ids_to_tokens(trg_indices[1:-1])

# === 加载测试集 ===
test_dataset = load_from_disk("test_dataset_processed")

# === 计算 BLEU 分数 ===
all_preds = []
all_targets = []
smoother = SmoothingFunction()

for example in tqdm(test_dataset, desc="翻译中", unit="句子"):
    src_text = example["translation"]["en"]
    ref_text = example["translation"]["de"]

    # 模型输出 token 序列
    pred_tokens = translate_to_tokens(src_text, tokenizer, model, device=device)

    # 参考翻译 token 序列
    ref_ids = tokenizer(ref_text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
    ref_tokens = tokenizer.convert_ids_to_tokens(ref_ids[1:-1])  # 去掉 BOS 和 EOS

    all_preds.append(pred_tokens)
    all_targets.append([ref_tokens])  # 注意：参考答案是二维结构

# === 用 NLTK 的 corpus_bleu 计算 BLEU-4 分数 ===
bleu_score = corpus_bleu(all_targets, all_preds,
                         smoothing_function=smoother.method1,
                         weights=(0.25, 0.25, 0.25, 0.25))

print(f"\n✅ 测试集 BLEU Score（nltk version）: {bleu_score:.4f}")

# === 打印翻译样例 ===
print("\n🌍 翻译示例（前 5 个）:")
for i in range(min(5, len(all_preds))):
    pred_sentence = tokenizer.convert_tokens_to_string(all_preds[i])
    target_sentence = tokenizer.convert_tokens_to_string(all_targets[i][0])
    print(f"\n🔹 原文：{test_dataset[i]['translation']['en']}")
    print(f"🔸 参考翻译：{target_sentence}")
    print(f"🔸 模型输出：{pred_sentence}")
