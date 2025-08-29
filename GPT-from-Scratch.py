import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import re

# --- 1. 參數設定 (Hyperparameters) ---
BATCH_SIZE = 64      # 一次處理多少個序列
BLOCK_SIZE = 256     # 序列的最大長度 (上下文長度)
MAX_ITERS = 5000     # 總共要訓練多少次
EVAL_INTERVAL = 500  # 每隔多少次評估一次模型效能
LEARNING_RATE = 3e-4 # 學習率
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 判斷用 GPU 還是 CPU
EVAL_ITERS = 200     # 評估時迭代的次數
N_EMBED = 384        # 嵌入維度 (每個詞彙/字元用多長的向量表示)
N_HEAD = 6           # 多頭注意力機制的「頭」數
N_LAYER = 6          # Transformer Block 的層數
DROPOUT = 0.2        # Dropout 比例，防止過擬合
MAX_DATASET_SIZE = 10000  # 限制資料集大小以加快訓練

print(f"將在 {DEVICE} 上進行訓練...")

# --- 2. 資料準備 (使用 CNN/DailyMail 資料集) ---
print("正在下載 CNN/DailyMail 資料集...")

def clean_text(text):
    """清理文本，移除多餘空格和特殊字符"""
    # 將多個空格替換為單個空格
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字符，保留基本標點符號
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    return text.strip()

def prepare_cnn_dailymail_data():
    """準備 CNN/DailyMail 資料集"""
    try:
        # 載入資料集 (只載入一小部分以加快速度)
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"train[:{MAX_DATASET_SIZE}]")
        
        print(f"成功載入 {len(dataset)} 篇文章")
        
        # 合併所有文章和摘要
        all_text = []
        for item in dataset:
            # 合併文章內容和摘要
            article = clean_text(item['article'])
            highlights = clean_text(item['highlights'])
            
            # 組合格式：文章 + 特殊分隔符 + 摘要
            combined_text = f"{article}\n[SUMMARY]\n{highlights}\n[END]\n"
            all_text.append(combined_text)
        
        # 將所有文本合併為一個大字串
        full_text = " ".join(all_text)
        
        print(f"處理完成，總文字長度: {len(full_text)} 字元")
        
        return full_text
        
    except Exception as e:
        print(f"載入資料集時發生錯誤: {e}")
        print("使用備用文本...")
        # 備用文本 - 一個較長的示例文本
        backup_text = """
        The field of artificial intelligence has seen remarkable progress in recent years. 
        Machine learning models, particularly large language models, have demonstrated 
        impressive capabilities in understanding and generating human-like text. These 
        models are trained on vast amounts of text data and learn to predict the next 
        word in a sequence. The transformer architecture, introduced in the paper 
        "Attention is All You Need", has become the foundation for many state-of-the-art 
        language models. The key innovation of transformers is the self-attention mechanism, 
        which allows the model to weigh the importance of different words in the input 
        when making predictions. This has led to significant improvements in natural 
        language processing tasks such as machine translation, text summarization, 
        and question answering. GPT (Generative Pre-trained Transformer) models have 
        shown particular promise in generating coherent and contextually relevant text. 
        These models are first pre-trained on large text corpora using a language modeling 
        objective, where they learn to predict the next token given the previous tokens. 
        After pre-training, these models can be fine-tuned for specific downstream tasks 
        or used for text generation through techniques like prompting. The success of 
        these models has sparked widespread interest in artificial intelligence and has 
        led to numerous applications in various domains including education, healthcare, 
        customer service, and creative writing.
        """ * 50  # 重複50次以確保有足夠的文本長度
        return backup_text

# 準備資料
text = prepare_cnn_dailymail_data()

# 確保文本長度足夠
if len(text) < BLOCK_SIZE + 100:  # 至少需要比 BLOCK_SIZE 大一些
    print(f"警告：文本長度 ({len(text)}) 太短，正在擴展...")
    text = text * (BLOCK_SIZE // len(text) + 2)  # 重複文本直到長度足夠

# 建立字元詞彙表
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"詞彙表大小: {vocab_size} 個字元")
print(f"文本長度: {len(text)} 字元")

# 建立字元到索引 (stoi) 和索引到字元 (itos) 的對應關係
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # 編碼器: 將字串轉為數字列表
decode = lambda l: ''.join([itos[i] for i in l]) # 解碼器: 將數字列表轉回字串

# 將所有文本資料轉換為 Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# 將資料分為訓練集和驗證集 (90% 訓練, 10% 驗證)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"訓練資料大小: {len(train_data)}")
print(f"驗證資料大小: {len(val_data)}")

# 資料載入器
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # 確保有足夠的資料進行採樣
    if len(data) <= BLOCK_SIZE:
        raise ValueError(f"資料長度 ({len(data)}) 必須大於 BLOCK_SIZE ({BLOCK_SIZE})")
    
    # 隨機選取 batch_size 個起始點
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    # 建立輸入序列 x 和目標序列 y
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    # y 是 x 的下一個字元
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# --- 3. 模型定義 (核心組件) ---
class Head(nn.Module):
    """ 單一注意力頭 """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        # tril 是一個下三角矩陣，用於遮蔽未來的資訊 (Decoder 的核心)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # 計算注意力分數 ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
        # 遮蔽未來的 token，使其不參與計算
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # 執行加權聚合
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ 多個注意力頭並行運算 """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # 將所有頭的結果串接起來
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ 一個簡單的前饋神經網路 """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Block: 結合了注意力和前饋網路 """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 殘差連接 (Residual Connection)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- 4. 完整的 GPT 模型 ---
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # token_embedding_table: 每個 token 對應到一個向量
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        # position_embedding_table: 每個位置對應到一個向量，讓模型學習位置資訊
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        # 組合多個 Transformer Block
        self.blocks = nn.Sequential(*[Block(N_EMBED, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBED) # 最後的 LayerNorm
        self.lm_head = nn.Linear(N_EMBED, vocab_size) # 線性層，將結果轉為詞彙表大小

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, N_EMBED)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T, N_EMBED)
        x = tok_emb + pos_emb # (B, T, N_EMBED)
        x = self.blocks(x) # (B, T, N_EMBED)
        x = self.ln_f(x) # (B, T, N_EMBED)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # 計算損失 (loss)，也就是評估模型預測的好壞
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Cross-entropy loss 是分類問題中常用的損失函數
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # 讓模型生成新的 token
        for _ in range(max_new_tokens):
            # 只看最後 block_size 的上下文
            idx_cond = idx[:, -BLOCK_SIZE:]
            # 取得預測結果 (logits)
            logits, loss = self(idx_cond)
            # 只關注最後一個時間步的結果
            logits = logits[:, -1, :] # (B, C)
            # 轉換為機率
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 從機率分佈中抽樣
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 將新的 token 加到序列中，繼續下一次生成
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# --- 5. 訓練流程 ---
model = GPTLanguageModel()
m = model.to(DEVICE)

# 計算模型參數數量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型參數總數: {total_params:,}")

# 建立 PyTorch 優化器
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 用於評估模型效能的函數
@torch.no_grad() # 告訴 PyTorch 在這個函數中不需要計算梯度
def estimate_loss():
    out = {}
    model.eval() # 將模型設為評估模式
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 將模型設回訓練模式
    return out

# 訓練迴圈
print("\n開始訓練...")
model.train() # 確保模型在訓練模式
for iter in range(MAX_ITERS):
    # 每隔一段時間評估一次模型
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        losses = estimate_loss()
        print(f"步驟 {iter}: 訓練損失 {losses['train']:.4f}, 驗證損失 {losses['val']:.4f}")

    # 取得一批訓練資料
    xb, yb = get_batch('train')

    # 評估損失
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) # 清除上一次的梯度
    loss.backward() # 反向傳播，計算梯度
    optimizer.step() # 更新模型參數

print("\n--- 訓練完成 ---")

# --- 6. 生成文本 ---
print("\n開始生成文本...\n")
# 給定一個起始 token (這裡用一個換行符作為開始)
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
# 讓模型生成 500 個 token
generated_text = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print("生成的文本：")
print("-" * 50)
print(generated_text)
print("-" * 50)

# 也可以嘗試給定一個特定的開始文本
start_text = "The news reported that"
start_tokens = torch.tensor([encode(start_text)], dtype=torch.long, device=DEVICE)
conditional_text = decode(m.generate(start_tokens, max_new_tokens=200)[0].tolist())
print(f"\n基於 '{start_text}' 生成的文本：")
print("-" * 50)
print(conditional_text)
print("-" * 50)
