---
title: "从白板到代码——图解 Decoder-only Transformer 的完整数学链路"
date: 2026-08-29
tags: [transformer, llm, attention, deep-learning]
draft: false
---

# 从白板到代码——图解 Decoder-only Transformer 的完整数学链路

## 为什么从 Decoder-only 入手

GPT、LLaMA、Qwen、ChatGLM……今天几乎所有能续写、能对话的大模型，内部都共享同一副骨架：Decoder-only Transformer。它并不神秘——「生成下一个 token」这件事，本质就是一次前向传播：一串 token id 走进去，一个概率分布走出来，再按分布抽一个字接在句尾，循环往复，就写出了整段话。

这篇文章只做一件事：把这条从「白板上的公式」到「可运行的 numpy/torch 代码」的数学链路，一段一段拆开。每一步都标注张量形状，每一步都回答「为什么这么设计」——为什么注意力要除以 √d_head 而不是 √d_model、为什么因果 mask 必须加在 softmax 之前、为什么 RoPE 能编码相对位置、为什么 LayerNorm 的方差分母是 N 而不是 N-1。读完你会得到一个能从头到尾跑通的小型前向实现，而不是一堆名词。

整条链路只有 5 段，几乎全是矩阵运算：token embedding 查表 → 多头自注意力 → Pre-Norm 残差块 → FFN → 输出投影 + softmax 采样。下面先给全景图。

## 全景图：一条前向链路的形状版图

沿用后文数值示例的配置（batch=2, seq=4, d_model=8, n_heads=2, d_head=4, vocab=16），每一步的形状标注在右侧：

```
token ids  (B, T)
   │  W_e[tokens]  查表
   ▼
embedding  (B, T, d_model)
   │
   ▼  ┌────────────────────────── 每层循环 ×L ──────────────────────────┐
      │  Pre-Norm:  x = LayerNorm(h)                 (B, T, d_model)  │
      │  Q,K,V = x@Wq, x@Wk, x@Wv                    (B, T, d_model)  │
      │  reshape→transpose                            (B, H, T, d_head)│
      │  Q,K 施加 RoPE（仅 Q/K，按 d_head 两两旋转）                      │
      │  scores = QKᵀ/√d_head + causal_mask          (B, H, T, T)     │
      │  attn = softmax(scores)                       (B, H, T, T)     │
      │  y = (attn@V) 拼回头 @Wo                       (B, T, d_model)  │
      │  h = h + y          ← 残差                                     │
      │  Pre-Norm:  x = LayerNorm(h)                 (B, T, d_model)  │
      │  h = h + GELU(x@W1)@W2                       (B, T, d_model)  │
      └────────────────────────────────────────────────────────────────┘
   │  最终 LayerNorm
   ▼
logits = h@W_lm                    (B, T, V)
   │  softmax(logits / τ)
   ▼
probs → 采样下一个 token
```

一个关键的不变量贯穿始终：d_model = n_heads × d_head。所有多头变换都把 d_model 拆成 heads 份再各自计算，最后拼回 d_model，维度从不错位。

## 第 1 步：Token Embedding 与位置信息

**查表嵌入。** 模型不认识字符，只认识整数 id。一张 embedding 表 W_e ∈ R^(V×d_model)（V 为词表大小），把每个 token id 映射成一条 d_model 维向量：

```python
h = W_e[tokens]        # tokens: (B, T) 整数 id → h: (B, T, d_model)
```

这一步是纯查表，没有任何计算，但它是整个模型的「词义入口」。实践中常有两个细节：原论文把 embedding 权重乘上 √d_model 以抵消其小方差，并让 W_e 与最后的输出投影共享权重（weight tying）；LLaMA 则不共享。对理解链路而言，记住「查表得到 (B,T,d_model)」就够。

**位置信息：为什么必须显式给。** 注意力本身对位置无感——把一句话里的词打乱顺序，attention 算出来的结果会完全一样（它是集合级别的加权和）。所以必须在 token 里注入「第几个位置」。经典方案是 sin/cos 绝对位置编码，直接**加**到 embedding 上：

```
PE(pos,2i)   = sin(pos / 10000^(2i/d_model))
PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))
```

现代 LLM（LLaMA 系）主流则用 RoPE（Rotary Position Embedding）。它不做加法，而是把 Q、K 向量按相邻两维一组做**二维旋转**，旋转角 = 位置 × 频率。设频率 θ_i = 10000^(-2i/d_head)，第 m 个位置、第 i 对维度的旋转变为：

```
[q0, q1] → [ q0·cos(mθ_i) − q1·sin(mθ_i),  q0·sin(mθ_i) + q1·cos(mθ_i) ]
```

写成 numpy 就是十几行：

```python
import numpy as np

def rope_freqs(seq_len, head_dim):
    # 频率 theta_i = 10000^(-2i/d_head)，i = 0..d_head/2-1（0 起始下标）
    theta = 10000.0 ** (-2.0 * np.arange(head_dim // 2) / head_dim)
    m = np.arange(seq_len).reshape(-1, 1)      # (T, 1) 各位置
    return np.cos(m * theta), np.sin(m * theta)  # 旋转角 = 位置 × 频率

def apply_rope(x, cos, sin):
    # x: (B, H, T, d_head)，相邻两维一组做 2D 旋转
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1)
```

**为什么 RoPE 能编码相对位置？** 这是它最精妙的一点。对第 m 个位置的 query 和第 n 个位置的 key，旋转后做点积：

```
(R_m·W_q·x_m)ᵀ · (R_n·W_k·x_n) = x_mᵀ W_qᵀ (R_mᵀ R_n) W_k x_n = x_mᵀ W_qᵀ R_{n−m} W_k x_n
```

旋转矩阵满足 R_mᵀ R_n = R_{n−m}，于是点积只依赖**相对距离 n−m**，而不是绝对位置。这就是「相对位置」的数学来源。三个易错点：RoPE 是乘法式而非加法式；只作用在 Q 和 K 上，V 不动；按 d_head（不是 d_model）两两分组旋转。

## 第 2 步：多头自注意力（Multi-Head Self-Attention）

注意力是整条链路的心脏。先把 d_model 维的隐状态投影成 Q、K、V，再拆成 H 个头：

```python
q = x @ W_q; k = x @ W_k; v = x @ W_v          # 各 (B, T, d_model)
# 关键顺序：先 view 成 (B,T,H,d_head)，再 transpose 到 (B,H,T,d_head)
q = q.reshape(B, T, H, d_head).transpose(0, 2, 1, 3)
k = k.reshape(B, T, H, d_head).transpose(0, 2, 1, 3)
v = v.reshape(B, T, H, d_head).transpose(0, 2, 1, 3)
q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)   # RoPE 只作用 Q/K
```

**reshape/transpose 顺序不能错**：必须先把 d_model 切成 (H, d_head) 再转置，否则头与维度的对应关系会被打乱。接着是核心的 scaled dot-product：

```python
def scaled_dot_product_attention(q, k, v, mask):
    d_head = q.shape[-1]
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_head)  # (B,H,T,T)
    scores = scores + mask               # 因果 mask 加在 softmax 之前
    attn = softmax(scores)               # 每行和 = 1
    return attn @ v                      # (B,H,T,d_head)
```

**为什么除以 √d_head 而不是 d_model？** 假设 q、k 各分量独立、均值 0 方差 1，那么点积 q·k 是 d_head 个乘积之和，方差会膨胀到 d_head。维度越大，scores 的绝对值越大，softmax 越容易饱和到梯度近乎消失的「one-hot」区。除以 √d_head 把方差拉回 1，softmax 才能待在梯度敏感区。原论文除的是 √d_k（d_k 就是每头的 Q/K 维度），PyTorch 的 scaled_dot_product_attention 默认 scale=1/√E（E 即 query 最后一维），三者说的都是 d_head。

**因果 mask：位置 i 不许看到未来。** decoder 是在「预测下一个字」，生成第 i 个 token 时绝不能偷看第 i+1 个。做法是把上三角（不含对角线）的 scores 置为 −∞：

```
mask = triu(ones(T,T), k=1) 置 -inf，对角线保留 0：

      j=0   j=1   j=2   j=3
i=0 [  0   -inf  -inf  -inf ]
i=1 [  0     0   -inf  -inf ]
i=2 [  0     0     0   -inf ]
i=3 [  0     0     0     0  ]
```

**为什么 mask 必须加在 softmax 之前？** 因为 softmax 里 exp(−∞)=0，−inf 的位置最终注意力权重严格等于 0，干净地「封死」未来信息；如果放到 softmax 之后去乘 0，掩掉的列反而会破坏「行和 = 1」的归一化。而且上三角不含对角线，意味着位置 i **可以**看到自己。数值上，手写 softmax 记得先做 max-subtraction：

```python
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)   # max-subtraction：防上/下溢
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
```

（torch 内部会自动做这步，手写时必须显式。）最后把各头结果拼回 d_model，再过一次输出投影：

```python
y = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, d_model)   # 拼回头
y = y @ W_o                                                  # 输出投影
```

用一个微型配置（B=2,T=4,H=2,d_head=4,权重 std=0.02）跑出来的注意力权重，最能看清因果 mask 的教学意义——第 0 行只有 1 个非零权重（只能看自己），第 1 行的权重在前两列间分配，第 3 行才均匀铺满全部 4 列：

```
L0 注意力权重 (B0, H0)：
[[1.      0.      0.      0.     ]
 [0.4999  0.5001  0.      0.     ]
 [0.3331  0.3338  0.3331  0.     ]
 [0.2511  0.2497  0.2501  0.2491 ]]
```

## 第 3 步：残差连接与 LayerNorm（Pre-Norm）

注意力算完，不能直接替换输入，而是**残差相加**。原论文（2017）用 Post-Norm：先过子层再归一化，即 `LayerNorm(x + Sublayer(x))`；现代 LLM 一律改用 Pre-Norm——先归一化子层的输入，残差支路内不归一化：

```python
# Pre-Norm：LayerNorm 在残差支路内部
h = h + attention(layer_norm(h))   # 先 LN 再注意力，再残差相加
h = h + ffn(layer_norm(h))         # 先 LN 再 FFN，再残差相加
```

Pre-Norm 收敛更稳、允许更深网络，是 LLaMA、GPT-2 之后的默认选择。LayerNorm 本身是白板级公式：

```
y = (x − E[x]) / √(Var[x] + ε) · γ + β
```

三个关键细节：均值和方差沿**最后一维**（特征维，不是 batch 维）计算；方差用**有偏估计**（分母 N，不是 N-1）；γ 初始化为 1、β 为 0，默认 ε=1e-5。

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mu  = x.mean(-1, keepdims=True)   # 沿最后一维求均值
    var = x.var(-1, keepdims=True)    # 有偏方差：分母 N（不是 N-1）
    return (x - mu) / np.sqrt(var + eps) * gamma + beta
```

这里分母用 N 而非 N-1 是有意为之——LayerNorm 追求的是把激活的尺度归一化，不是做统计学的无偏估计；PyTorch 里 `torch.var(x, correction=0)` 与之一致。顺带一提，LLaMA 系连这个都用更省算力的 RMSNorm 替代（只除以均方根，不含去均值），但数学精神相同。

## 第 4 步：FFN（前馈网络）

注意力负责「在位置之间搬运信息」，FFN 则负责「在每个位置上独立做非线性变换」——它让模型具备逐 token 的存储与加工能力。原论文用 ReLU，GPT-2 之后普遍用 GELU，LLaMA 系改用 SwiGLU。

**GELU（本文示例采用）。** 精确形式是 x 乘以标准正态分布的 CDF：

```
GELU(x) = x·Φ(x) = 0.5·x·(1 + erf(x/√2))
```

```python
import math
def gelu_exact(x):
    # GELU(x) = x · Φ(x)，Φ 是标准正态 CDF（精确 erf 形式）
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))
```

工程上常用 tanh 近似以省去 erf：`GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`。FFN 先把 d_model 扩到 4d_model 再压回来：

```python
h = h + gelu_exact(layer_norm(h) @ W1) @ W2    # W1:(d_model,4d) W2:(4d,d_model)
```

**SwiGLU（现代主流）。** 它把「门控」引入 FFN：两个线性投影做逐元素相乘，其中一个先过 SiLU 激活，再乘第三个矩阵投影回 d_model：

```
FFN(x) = W2(SiLU(W1·x) ⊙ (W3·x)),   d_hidden = (2/3)·4d
```

```python
# SwiGLU（LLaMA 系）：三权重门控，中间维取 2/3·4d 而非 4d
h = w2(silu(x @ w1) * (x @ w3))      # SiLU(z) = z·σ(z)
```

注意两点：SwiGLU 是**三**个权重（w1/w2/w3）而非两个；因为门控多了一份中间激活，LLaMA 把中间维从 PaLM 的 4d 缩到 (2/3)·4d，以保持总参数量相当。用 GELU 的旧式 FFN 中间维仍是 4d。

## 第 5 步：输出投影与采样

跑完 L 层残差块后，做最后一次 LayerNorm，然后把 d_model 维投影回词表大小，得到每个位置对 V 个 token 的原始打分 logits：

```python
h = layer_norm(h)
logits = h @ W_lm                          # (B, T, V)
probs  = softmax(logits / temperature)     # temperature 在 softmax 之前除
next_id = np.random.choice(V, p=probs[0, -1])   # 只取最后一个位置的分布采样
```

采样前可先除以 temperature：τ<1 让分布更尖锐（更「确定」），τ>1 更平缓（更「发散」），τ=1 即不缩放。注意 temperature 作用在 softmax **之前**，不是之后。解码用 categorical 采样（`torch.multinomial` 或 `np.random.choice`）而非 argmax，是为了让同一句 prompt 能长出不同的下文——这是生成式模型区别于「查表」的本质。采样是随机的：同一种子下，本示例一次运行里两个样本的最后一位抽到了 `[8, 11]`，但真正有意义的是 probs 那个 16 维分布，单次抽取本身没有信息量。

## 收尾：把它们拼成完整的 forward

把上面的片段收拢，一个能跑通、形状全程对齐的 Decoder-only 完整 forward 约 55 行：

```python
import math, numpy as np

B, T = 2, 4
D, H, DH = 8, 2, 4          # d_model, n_heads, d_head
V, FFN, L = 16, 32, 2       # vocab, FFN 中间维, 层数
EPS = 1e-5

rng = np.random.default_rng(0)                    # 固定种子：权重与 tokens 完全可复现
W_e = rng.normal(0, 0.02, (V, D))                 # token embedding 表
W_q = [rng.normal(0, 0.02, (D, D)) for _ in range(L)]
W_k = [rng.normal(0, 0.02, (D, D)) for _ in range(L)]
W_v = [rng.normal(0, 0.02, (D, D)) for _ in range(L)]
W_o = [rng.normal(0, 0.02, (D, D)) for _ in range(L)]
W1  = [rng.normal(0, 0.02, (D, FFN)) for _ in range(L)]
W2  = [rng.normal(0, 0.02, (FFN, D)) for _ in range(L)]
gamma = np.ones(D); beta = np.zeros(D)            # LayerNorm 参数（演示用，各层共享）
W_lm = rng.normal(0, 0.02, (D, V))                # 输出投影（未做 weight tying，简化）
tokens = rng.integers(0, V, (B, T))               # 输入 token id (2,4)

def layer_norm(x, gamma, beta, eps=EPS):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)  # 分母 N
    return (x - mu) / np.sqrt(var + eps) * gamma + beta

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)          # max-subtraction
    e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)

def gelu_exact(x): return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))

def rope_freqs(seq_len, head_dim):
    theta = 10000.0 ** (-2.0 * np.arange(head_dim // 2) / head_dim)
    m = np.arange(seq_len).reshape(-1, 1)
    return np.cos(m * theta), np.sin(m * theta)

def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2; x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1*cos[None,None] - x2*sin[None,None],
                           x1*sin[None,None] + x2*cos[None,None]], axis=-1)

mask = np.where(np.triu(np.ones((T, T)), k=1) == 1, -np.inf, 0.0)  # 上三角 -inf
cos, sin = rope_freqs(T, DH)

def forward(tokens):
    h = W_e[tokens]                                    # (B,T,D)
    for l in range(L):
        x = layer_norm(h, gamma, beta)                 # Pre-Norm ① 注意力
        q = apply_rope((x @ W_q[l]).reshape(B,T,H,DH).transpose(0,2,1,3), cos, sin)
        k = apply_rope((x @ W_k[l]).reshape(B,T,H,DH).transpose(0,2,1,3), cos, sin)
        v = (x @ W_v[l]).reshape(B,T,H,DH).transpose(0,2,1,3)
        a = softmax(q @ k.transpose(0,1,3,2) / np.sqrt(DH) + mask)  # mask 在 softmax 前
        h = h + (a @ v).transpose(0,2,1,3).reshape(B,T,D) @ W_o[l]  # 残差
        h = h + gelu_exact(layer_norm(h, gamma, beta) @ W1[l]) @ W2[l]  # FFN 残差
    return softmax(layer_norm(h, gamma, beta) @ W_lm)   # (B,T,V) 概率

probs = forward(tokens)
```

对着这段代码回头看四件验证过的事：softmax 每一行和恒等于 1；因果 mask 的 10 个非 −inf 位置全部满足「列 ≤ 行」；LayerNorm 输出均值实测 −2.78e-17、方差 1.00；attention 输出与逐位置手算的参考实现逐元素一致（atol=1e-10）。公式与代码，就这样在同一张白板上对上了。

## 参考阅读

- Attention Is All You Need (Vaswani et al., 2017) — https://arxiv.org/abs/1706.03762
- RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021) — https://arxiv.org/abs/2104.09864
- LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023) — https://arxiv.org/abs/2302.13971
- GLU Variants Improve Transformer (Shazeer, 2020) — https://arxiv.org/abs/2002.05202
- RMSNorm (Zhang & Sennrich, 2019) — https://arxiv.org/abs/1910.07467
- Karpathy nanoGPT — https://github.com/karpathy/nanoGPT/blob/master/model.py
- LLaMA 参考实现 — https://github.com/facebookresearch/llama/blob/main/llama/model.py
