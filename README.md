# GPT-from-Scratch
Implementing a GPT Language Model from Scratch. This project uses PyTorch to train a character-level language model, explaining and demonstrating the core principles of Transformers and self-attention mechanisms.

這是一個教學專案，旨在從零開始建構一個簡易版的 GPT (Generative Pre-trained Transformer) 語言模型。我們將使用 **PyTorch** 框架，並專注於理解 Transformer 架構的核心組件，如自注意力機制 (Self-Attention)。

### 🧠 核心程式碼解析

本專案的核心是 `GPTLanguageModel` 類別，它由以下幾個關鍵組件構成：

#### **1. 自注意力頭 (Self-Attention Head)**
- 類別：`Head`
- 功能：這是 Transformer 的核心。它根據輸入序列中的 **Query**、**Key** 和 **Value** 向量，計算每個字元對其他字元的關聯性，並進行加權聚合。
- 關鍵技巧：使用了 **下三角矩陣 (tril)** 來遮蔽未來的資訊，確保模型在預測時只考慮過去的上下文。

#### **2. 多頭注意力 (Multi-Head Attention)**
- 類別：`MultiHeadAttention`
- 功能：將多個自注意力頭的結果串接起來。每個頭能從不同角度學習輸入的關係，讓模型能更全面地理解上下文。

#### **3. Transformer Block**
- 類別：`Block`
- 功能：一個完整的 Transformer 層，由多頭注意力、前饋神經網路 (FeedForward)、層正規化 (LayerNorm) 和 **殘差連接 (Residual Connection)** 組成。殘差連接有助於解決梯度消失問題，讓深層模型能順利訓練。

#### **4. GPT 模型**
- 類別：`GPTLanguageModel`
- 功能：將多個 `Block` 層疊加，並結合 **詞嵌入** (Token Embeddings) 和 **位置嵌入** (Positional Embeddings)。這使得模型不僅能學會詞彙的語意，也能學習它們在序列中的位置和順序。
