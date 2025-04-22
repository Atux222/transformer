import torch
from transformer_first import Transformer
from transformers import AutoTokenizer
import sacrebleu

# === 加载模型与 Tokenizer ===
checkpoint_file = "/home/autolab/xyy/checkpoints/checkpoint_epoch_4.pth"
model_path = "/home/autolab/xyy/token_model"
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

# 创建模型
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

# === 翻译函数 ===
def translate_sentence(sentence, tokenizer, model, device, max_len=128):
    model.eval()
    with torch.no_grad():
        # 编码输入
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        src = inputs["input_ids"].to(device)

        # 初始解码器输入：[CLS] 作为 <bos>
        trg_indices = [tokenizer.cls_token_id]

        for _ in range(max_len):
            trg_tensor = torch.tensor([trg_indices], dtype=torch.long).to(device)
            output = model(src, trg_tensor)  # (1, seq_len, vocab_size)
            next_token = output[0, -1, :].argmax(dim=-1).item()
            trg_indices.append(next_token)
            if next_token == tokenizer.sep_token_id:  # 遇到 <eos> 则停止
                break

        # 转换为字符串（去除 <bos> 和 <eos>）
        tokens = tokenizer.convert_ids_to_tokens(trg_indices[1:-1])
        translated_text = tokenizer.convert_tokens_to_string(tokens)
        return translated_text.strip()

# === 示例用法 ===
if __name__ == "__main__":
    English_sentence = "The scientists who have been working on this complicated research project for years hope that their discoveries can make a significant contribution to solving global environmental problems."
    german_translation = translate_sentence(English_sentence, tokenizer, model, device=device)
    print("英语：", English_sentence)
    print("翻译：", german_translation)
