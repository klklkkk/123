import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel

# =========================
# 配置部分（可自行根据需要修改）
# =========================
DATA_FOLDER = "data"  # 存放 jpg 和 txt 的文件夹
TRAIN_FILE = "train.txt"
TEST_FILE = "test_without_label.txt"
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
MAX_TEXT_LENGTH = 64  # BERT 输入的最大文本长度
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 这里指定从本地的 "bert-base-uncased" 文件夹读取模型
LOCAL_BERT_FOLDER = "./bert-base-uncased"

# 设定随机种子，保证可复现
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(RANDOM_SEED)

# =========================
# 一、读取训练数据（guid, label）并划分训练集/验证集
# =========================
guid_list = []
label_list = []
label_map = {"positive": 0, "neutral": 1, "negative": 2}

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头 guid,tag
    for row in reader:
        guid, tag = row
        guid_list.append(guid)
        label_list.append(label_map[tag])

all_data = list(zip(guid_list, label_list))
random.shuffle(all_data)

# 简单地 8:2 进行训练集、验证集划分
train_ratio = 0.8
train_size = int(train_ratio * len(all_data))
val_size = len(all_data) - train_size
train_data = all_data[:train_size]
val_data = all_data[train_size:]

# =========================
# 二、定义数据集类，读取图像 + 文本
# =========================
class MultiModalDataset(Dataset):
    def __init__(self, data_list, data_folder, tokenizer, max_len=64):
        """
        data_list: [(guid, label), ...]
        """
        self.data_list = data_list
        self.data_folder = data_folder
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        guid, label = self.data_list[idx]
        # 读取文本
        txt_path = os.path.join(self.data_folder, f"{guid}.txt")
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        # 读取图像
        img_path = os.path.join(self.data_folder, f"{guid}.jpg")
        image = Image.open(img_path).convert("RGB")

        # 文本tokenize
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)        # (max_len)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (max_len)

        # 图像变换
        image = self.transform(image)  # (3, 224, 224)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "guid": guid
        }

# =========================
# 三、构建模型(文本BERT + 图像ResNet)，再融合
# =========================
class MultiModalModel(nn.Module):
    def __init__(self, num_labels=3):
        super(MultiModalModel, self).__init__()
        # 文本模型: 从本地预训练的BERT读取
        self.bert = BertModel.from_pretrained(LOCAL_BERT_FOLDER)
        self.text_fc = nn.Linear(self.bert.config.hidden_size, 256)

        # 图像模型: ResNet (使用预训练权重)
        self.resnet = models.resnet50(pretrained=True)
        # 替换最后一层全连接
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 256)

        # 融合后的全连接
        self.classifier = nn.Linear(256 + 256, num_labels)

    def forward(self, input_ids, attention_mask, images):
        # 1. 文本特征
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_outputs.last_hidden_state[:, 0, :]  # 取CLS向量
        text_embed = self.text_fc(text_cls)                 # 映射到256维

        # 2. 图像特征
        img_embed = self.resnet(images)                     # 256维

        # 3. 拼接融合
        fusion = torch.cat((text_embed, img_embed), dim=1)  # [batch_size, 512]
        logits = self.classifier(fusion)                    # [batch_size, num_labels]
        return logits

# =========================
# 四、初始化数据集与DataLoader
# =========================
# 从本地目录读取tokenizer
tokenizer = BertTokenizer.from_pretrained(LOCAL_BERT_FOLDER)

train_dataset = MultiModalDataset(train_data, DATA_FOLDER, tokenizer, max_len=MAX_TEXT_LENGTH)
val_dataset = MultiModalDataset(val_data, DATA_FOLDER, tokenizer, max_len=MAX_TEXT_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# 五、训练模型
# =========================
model = MultiModalModel(num_labels=3).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f} "
          f"Val Loss: {val_loss:.4f} "
          f"Val Acc: {val_acc:.4f}")

# =========================
# 六、在测试集上预测输出结果
# =========================
test_data = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头 guid,tag
    for row in reader:
        guid, _ = row
        # 暂用一个假label占位(不会用到), 方便Dataset处理
        test_data.append((guid, 0))

test_dataset = MultiModalDataset(test_data, DATA_FOLDER, tokenizer, max_len=MAX_TEXT_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
id2label = {v: k for k, v in label_map.items()}
predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        images = batch["image"].to(DEVICE)
        guids = batch["guid"]
        logits = model(input_ids, attention_mask, images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        for guid, p in zip(guids, preds):
            predictions.append((guid, id2label[p]))

# =========================
# 七、将预测结果输出到文件
# =========================
with open("predict_result.txt", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["guid", "tag"])
    for guid, tag in predictions:
        writer.writerow([guid, tag])

print("预测完成！结果已写入 predict_result.txt。")
