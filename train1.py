import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import wandb
from transformer_first import Transformer
from dataloader import load_or_process_dataset
from transformers import get_scheduler, AutoModel, AutoTokenizer
import os

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 模型路径
model_path = "/home/autolab/xyy/token_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 超参数
d_model = 512
d_ff = 2048
d_k = d_v = 64
n_layers = 6
n_heads = 8
drop_prob = 0.1
max_length = 128
batch_size = 16
num_epochs = 4
learning_rate = 4e-5

# 数据加载
train_dataset, valid_dataset, test_dataset = load_or_process_dataset(max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
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

# 损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)



# checkpoint 路径
checkpoint_path = "checkpoints"
checkpoint_file = os.path.join(checkpoint_path, "best_checkpoint.pth")
os.makedirs(checkpoint_path, exist_ok=True)

# 是否加载 checkpoint
start_epoch = 0
best_loss = float("inf")
if os.path.exists(checkpoint_file):
    print(f"Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint["loss"]
    print(f"Resuming from epoch {start_epoch} with loss {best_loss:.4f}")
else:
    print("No checkpoint found. Training from scratch.")

# wandb 初始化
wandb.init(project="transformer-training", name="run-1", save_code=True)
wandb.watch(model, log="all")

# 训练
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        enc_inputs = torch.stack(batch["input_ids"],dim=0).to(device)# [batch_size, src_len] # shape = torch.Size([128, 32])
        enc_inputs = enc_inputs.transpose(0, 1)  # [128, 32] → [32, 128]

        dec_inputs = torch.stack(batch["labels"],dim=0).to(device)  
        dec_inputs = dec_inputs.transpose(0, 1)
        dec_inputs = dec_inputs[:, :-1]

        dec_outputs = torch.stack(batch["labels"],dim=0).to(device) 
        dec_outputs = dec_outputs.transpose(0, 1)
        dec_outputs = dec_outputs[:, 1:]  # [batch_size, seq_len-1]

        outputs = model(enc_inputs, dec_inputs)
        outputs = outputs.reshape(-1, outputs.size(-1))
        dec_outputs = dec_outputs.reshape(-1)

        loss = criterion(outputs, dec_outputs)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100000 == 0 and batch_idx != 0:
            checkpoint_batch = {
                "epoch": epoch + 1,
                "batch_idx": batch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item()
            }
            filename = f"{checkpoint_path}/checkpoint_epoch{epoch+1}_batch{batch_idx}.pth"
            torch.save(checkpoint_batch, filename)
            print(f"[Epoch {epoch+1} | Batch {batch_idx}] Saved periodic checkpoint to {filename}")
            wandb.log({"periodic_batch_loss": loss.item(), "epoch": epoch + 1, "batch_idx": batch_idx})


        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
            wandb.log({"epoch": epoch + 1, "batch_loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }
    torch.save(checkpoint, f"{checkpoint_path}/checkpoint_epoch_{epoch+1}.pth")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(checkpoint, f"{checkpoint_path}/best_checkpoint.pth")

print("训练完成！")
