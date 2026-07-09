# finger-ml: Mathematical Model and Algorithm Pipeline

This document describes the offline hand-gesture **event detection** system implemented in `finger-ml`: the feature construction, neural architecture, training objective, inference, and post-processin[...]

## 1. Problem Formulation

Given a recorded RGB video of a single hand performing discrete gestures, the system outputs a list of **events**:

$$
\mathcal{E} = \{(c_k, t_k^{\text{start}}, t_k^{\text{end}})\}_{k=1}^{K}
$$

where $c_k \in \{0,\ldots,6\}$ is a gesture class (6 gestures + background) and $t_k^{\text{start}}, t_k^{\text{end}}$ are frame indices (0-based in the implementation) or derived timestamps in mi[...]

This is cast as **dense temporal segmentation**: predict a class label $y_t$ for every frame $t = 0,\ldots,T-1$, then convert contiguous non-background segments into events.

| Label | Key | Meaning |
| --- | --- | --- |
| 0 | `pinch_index` | Thumb pinches index fingertip |
| 1 | `pinch_middle` | Thumb pinches middle fingertip |
| 2–5 | `thumb_slide_*` | Thumb slides up/down/left/right |
| 6 | `background` | Rest / transition |

---

## 2. End-to-End Pipeline

```mermaid
flowchart LR
    A[MP4 Video] --> B[MediaPipe Hand Landmarker\nVIDEO mode]
    B --> C[21-point landmarks]
    C --> D[Palm-local normalization]
    D --> E[Motion / contact features\nT x 21 x 12]
    E --> F[ST-GCN Encoder]
    F --> G[MS-TCN Temporal Head]
    G --> H[Frame logits + boundary head]
    H --> I[Overlapping chunk fusion]
    I --> J[Post-process\nsmooth / threshold / merge]
    J --> K[Events JSON]
```

### Stage A — Data collection (optional, for training)

`finger-collect` records MP4 + per-gesture start/end annotations in JSON.

### Stage B — Preprocessing (training only)

`finger-preprocess` runs MediaPipe on labeled videos and writes `data/features/<session>.npz`:

| Array | Shape | Description |
| --- | --- | --- |
| `landmarks` | `[T, 21, 3]` | Normalized skeleton |
| `features` | `[T, 21, 12]` | Extended per-node features |
| `labels` | `[T]` | Frame-level class (0–6) |
| `train_mask` | `[T]` | `False` on ignored transition frames |

### Stage C — Training

`finger-train` learns frame-wise class logits and boundary probabilities from `.npz` chunks.

### Stage D — Detection (inference)

`finger-detect` applies the same feature pipeline to any MP4 and writes event JSON (+ optional overlay video).

---

## 3. Hand Landmark Extraction

MediaPipe Hand Landmarker (VIDEO mode) outputs 21 normalized 3D points per frame:

$$
\mathbf{p}_t^{(i)} = (x, y, z)_t^{(i)}, \quad i = 0,\ldots,20
$$

If detection fails at frame $t$, the previous valid landmarks are held (zero-order hold).

---

## 4. Palm-Local Coordinate Normalization

Raw landmarks are converted to a **scale- and rotation-invariant** palm frame (`features.normalize_landmarks`).

1. **Translate** to wrist origin (index 0):
   $$
   \tilde{\mathbf{p}}_t^{(i)} = \mathbf{p}_t^{(i)} - \mathbf{p}_t^{(0)}
   $$

2. **Set scale** from index-MCP distance (index 5):
   $$
   s_t = \left\lVert \tilde{\mathbf{p}}_t^{(5)} \right\rVert_2
   $$
   (if $s_t < 10^{-6}$, fall back to $s_t = 1$ and $\mathbf{x} = (1,0,0)$).

3. **Build orthonormal basis** (note: pinky MCP uses the **centered**, unscaled vector):
   $$
   \mathbf{x} = \frac{\tilde{\mathbf{p}}_t^{(5)}}{s_t}, \quad
   \mathbf{z} = \frac{\mathbf{x} \times \tilde{\mathbf{p}}_t^{(17)}}{\left\lVert \mathbf{x} \times \tilde{\mathbf{p}}_t^{(17)} \right\rVert_2}, \quad
   \mathbf{y} = \frac{\mathbf{z} \times \mathbf{x}}{\left\lVert \mathbf{z} \times \mathbf{x} \right\rVert_2}
   $$

4. **Project** into palm frame with $B_t = [\mathbf{x}\ \mathbf{y}\ \mathbf{z}] \in \mathbb{R}^{3 \times 3}$ (columns = basis vectors):
   $$
   \mathbf{q}_t^{(i)} = \frac{\tilde{\mathbf{p}}_t^{(i)\top} B_t}{s_t}
   $$
   Equivalently, in code: `(centered @ basis) / scale`.

Output: $\mathbf{Q}_t \in \mathbb{R}^{21 \times 3}$.

---

## 5. Per-Node Feature Vector (12 channels)

For each frame, `build_motion_features` constructs $\mathbf{X}_t \in \mathbb{R}^{21 \times 12}$.

**Per-joint channels (0–5):**

| Channel | Symbol | Definition |
| --- | --- | --- |
| 0–2 | $x,y,z$ | Normalized joint position $\mathbf{q}_t^{(i)}$ |
| 3–5 | $dx,dy,dz$ | Temporal difference $\mathbf{q}_t^{(i)} - \mathbf{q}_{t-1}^{(i)}$ (zero at $t=0$) |

**Global channels (6–11), broadcast to all 21 joints:**

| Channel | Symbol | Definition |
| --- | --- | --- |
| 6 | $d_{TI}$ | $\left\lVert \mathbf{q}_t^{(4)} - \mathbf{q}_t^{(8)} \right\rVert_2$ (thumb tip – index tip) |
| 7 | $d_{TM}$ | $\left\lVert \mathbf{q}_t^{(4)} - \mathbf{q}_t^{(12)} \right\rVert_2$ (thumb tip – middle tip) |
| 8–10 | $\Delta_{TI}$ | $\mathbf{q}_t^{(4)} - \mathbf{q}_t^{(8)}$ (thumb–index displacement vector) |
| 11 | valid | MediaPipe detection flag $v_t \in \{0,1\}$, repeated across joints |

Tensor layout for the network: $\mathbf{X} \in \mathbb{R}^{B \times C \times T \times V}$ with $C=12$, $V=21$ (channels-first, as `x.permute(2, 0, 1)`).

---

## 6. Neural Architecture: Adaptive ST-GCN + MS-TCN

### 6.1 Graph Structure

Anatomical hand edges $\mathcal{E}$ (plus self-loops) define an adjacency matrix. The implementation initializes $\mathbf{A} = \mathbf{I}$, then sets $\mathbf{A}_{ij}=\mathbf{A}_{ji}=1$ for each[...]

$$
\hat{\mathbf{A}} = \mathbf{D}^{-1/2}\,\mathbf{A}\,\mathbf{D}^{-1/2}, \quad D_{ii} = \sum_j A_{ij}
$$

An **adaptive residual graph** is learned on top of the fixed normalized adjacency $\hat{\mathbf{A}}$:

$$
\mathbf{A}_{\text{adapt}} = \hat{\mathbf{A}} + 0.25 \cdot \tanh(\mathbf{A}_{\text{res}})
$$

### 6.2 Adaptive Graph Convolution

For input $\mathbf{X} \in \mathbb{R}^{B \times C_{\text{in}} \times T \times V}$:

$$
\mathbf{X}' = \text{Conv}_{1\times1}(\mathbf{X}), \qquad
\mathbf{Y}_{b,c,t,v} = \sum_{v'} \mathbf{X}'_{b,c,t,v'}\,\mathbf{A}_{\text{adapt},\,v',v}
$$

### 6.3 ST-GCN Block

Each block applies graph convolution + temporal convolution (kernel 9 along time) with batch norm, ReLU, dropout, and a residual connection:

$$
\text{STGCN}(\mathbf{X}) = \text{ReLU}\big(\text{TCN}(\text{GCN}(\mathbf{X})) + \text{Res}(\mathbf{X})\big)
$$

Encoder stack: $12 \to 48 \to 96 \to 128$ channels, then **global mean pool over joints**:

$$
\mathbf{H} \in \mathbb{R}^{B \times 128 \times T}
$$

### 6.4 MS-TCN-Style Temporal Head

Default: 2 temporal stages (`temporal_stages=2`) and 6 dilated residual layers per stage (dilations $2^i$: $1,2,4,8,16,32$).

**Stage 0** (input = encoder output $\mathbf{H}$):
$$
\mathbf{Z}^{(0)},\ \mathbf{L}^{(0)} = \text{TemporalStage}_0(\mathbf{H}), \quad
\mathbf{L}^{(0)} \in \mathbb{R}^{B \times 7 \times T}
$$
where $\mathbf{Z}^{(0)}$ is the hidden temporal feature map and $\mathbf{L}^{(0)}$ is the class logit map.

**Refinement stages** $s = 1,\ldots,S-1$ (default $S=2$, one refiner):
$$
\mathbf{L}^{(s)} = \text{Classifier}\big(\text{TemporalStage}_s(\text{softmax}(\mathbf{L}^{(s-1)}))\big)
$$

Final frame logits returned by the model: $\mathbf{L} = \mathbf{L}^{(S-1)}$. All stage logits $\{\mathbf{L}^{(s)}\}_{s=0}^{S-1}$ are supervised during training.

### 6.5 Boundary Head

From stage-0 hidden features $\mathbf{Z}^{(0)} \in \mathbb{R}^{B \times D \times T}$ (not from refined logits):

$$
\mathbf{B} = \text{Conv1D}_{2}\!\left(\text{ReLU}\!\left(\text{Conv1D}_{3\times1}(\mathbf{Z}^{(0)})\right)\right) \in \mathbb{R}^{B \times 2 \times T}
$$

Channel 0 = start-boundary logit, channel 1 = end-boundary logit.

---

## 7. Training Objective

For each chunk, let $\mathbf{L}^{(s)}$ be stage logits, $\mathbf{B}$ boundary logits, $\mathbf{y}$ frame labels, and $\mathbf{m} \in \{0,1\}^T$ the train mask (`train_mask`; ignored frames use[...]

### 7.1 Weighted Cross-Entropy (per stage)

Class weights balance rarity ($C=7$ classes):

$$
\tilde{w}_c = \frac{1}{\sqrt{n_c}}, \qquad
w_c = \frac{\tilde{w}_c}{\sum_{c'} \tilde{w}_{c'}} \cdot C, \quad n_c = \text{masked frame count of class } c
$$

$$
\mathcal{L}_{\text{CE}} = \frac{1}{S} \sum_{s=0}^{S-1} \text{CE}\big(\mathbf{L}^{(s)},\, \mathbf{y};\, \mathbf{w},\, \text{ignore}=-100\big)
$$

### 7.2 Boundary BCE

Binary targets $\mathbf{b} \in \{0,1\}^{2 \times T}$ mark gesture start/end transitions, with $\pm r$ frame radius (default $r=2$):

$$
\mathcal{L}_{\text{bound}} = \frac{1}{|\mathcal{M}|} \sum_{(k,t) \in \mathcal{M}} \text{BCEWithLogits}\big(B_{k,t},\, b_{k,t};\, \text{pos\_weight}=8\big)
$$

where $\mathcal{M} = \{(k,t) : m_t = 1\}$ restricts loss to trainable frames.

### 7.3 Temporal MSE (TMSE) Smoothness

Applied to the **final** stage logits $\mathbf{L}$. Uses `log_softmax` with stop-gradient on the previous frame:

$$
\Delta_{c,t} = \log p_{c,t} - \log p_{c,t-1}^{\;\text{(detach)}}, \quad \mathbf{p}_t = \text{softmax}(\mathbf{L}_t)
$$

$$
\mathcal{L}_{\text{TMSE}} = \frac{1}{|\mathcal{V}|} \sum_{(c,t) \in \mathcal{V}} \min\!\left(\Delta_{c,t}^{2},\; \tau^2\right), \quad \tau = 4
$$

where $\mathcal{V} = \{(c,t) : m_t = 1 \land m_{t-1} = 1\}$.

### 7.4 Total Loss

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_b \mathcal{L}_{\text{bound}} + \lambda_s \mathcal{L}_{\text{TMSE}}
$$

Defaults: $\lambda_b = 0.2$ (CLI default) or $0.3$ (trained checkpoint), $\lambda_s = 0.15$.

Optimizer: AdamW ($\text{lr}=3\times10^{-4}$, `weight_decay=1e-4`), ReduceLROnPlateau on validation frame accuracy (patience 6, factor 0.5). Gradient clipping at norm 5.0.

---

## 8. Inference: Overlapping Chunk Fusion

Long videos are processed in chunks of length $L$ (default 512) with overlap $O$ (default 128), stride $\text{stride} = \max(1,\, L - O)$.

For each chunk start index $s_0$ with valid length $\ell = \min(L,\, T - s_0)$:

$$
\bar{\mathbf{p}}_{s_0:s_0+\ell} \mathrel{+}= \text{softmax}\big(\mathbf{L}_{:,0:\ell}\big)^\top, \qquad
\bar{\mathbf{b}}_{:,s_0:s_0+\ell} \mathrel{+}= \sigma\big(\mathbf{B}_{:,0:\ell}\big)
$$

Short chunks are zero-padded to length $L$ before inference; only the first $\ell$ outputs are accumulated. After all chunks:

$$
\hat{\mathbf{p}}_t = \frac{\bar{\mathbf{p}}_t}{N_t}, \qquad
\hat{\mathbf{b}}_t = \frac{\bar{\mathbf{b}}_t}{N_t}
$$

where $N_t$ is the number of overlapping chunk predictions covering frame $t$.

---

## 9. Post-Processing: Frame Labels → Events

Given $\hat{\mathbf{p}}_t \in \mathbb{R}^7$:

1. **Argmax + confidence gate:**
   $$
   \tilde{y}_t = \arg\max_c \hat{p}_{t,c}, \qquad
   \hat{y}_t = \begin{cases}
   \tilde{y}_t & \text{if } \hat{p}_{t,\tilde{y}_t} \geq \theta \\
   6 \text{ (background)} & \text{otherwise}
   \end{cases}
   $$
   Default $\theta = 0.55$.

2. **Weighted temporal smoothing** (window $w=7$, half-width $\lfloor w/2 \rfloor = 3$): for each $t$, among non-background labels in the gated sequence $\hat{y}$ within $[t-3,\, t+3]$, pick[...]

3. **Segment extraction:** contiguous runs of the same non-background label with
   $$
   \text{duration}(t_i \ldots t_j) = j - i + 1 \geq L_{\min}, \quad L_{\min} = \max\!\left(1,\, \left\lfloor \frac{t_{\min}^{\text{ms}} \cdot \text{fps}}{1000} \right\rceil \right)
   $$
   Default $t_{\min}^{\text{ms}} = 120$.

4. **Boundary refinement:** for coarse segment $[i,j]$, set
   $\text{pad} = \max(3,\, \min(12,\, \lfloor(j-i+1)/2\rfloor))$. Search windows are clamped to $[0,\, T)$:
   $$
   s_0 = \max(0,\, i - \text{pad}),\ s_1 = \min(T,\, i + \text{pad} + 1),\quad
   i^* = s_0 + \arg\max_{t \in [s_0,\, s_1)} \hat{b}_{0,t}
   $$
   $$
   e_0 = \max(0,\, j - \text{pad}),\ e_1 = \min(T,\, j + \text{pad} + 1),\quad
   j^* = e_0 + \arg\max_{t \in [e_0,\, e_1)} \hat{b}_{1,t}
   $$
   (if $j^* < i^*$, set $j^* = i^*$).

5. **Merge:** combine adjacent same-class segments if frame gap $g = \text{start}_{k+1} - \text{end}_k - 1 \leq G_{\max}$, where $G_{\max} = \lfloor t_{\text{gap}}^{\text{ms}} \cdot \text{fps}[...]

Output event (0-based inclusive frame indices):
$$
\text{event}_k = \{\text{gesture},\, \text{start\_frame},\, \text{end\_frame},\, \text{start\_ms},\, \text{end\_ms},\, \text{mean\_conf}\}
$$
with $\text{start\_ms} = \lfloor 1000 \cdot \text{start\_frame} / \text{fps} \rceil$ and $\text{mean\_conf} = \frac{1}{j^* - i^* + 1}\sum_{t=i^*}^{j^*} \hat{p}_{t,c}$.

---

## 10. Evaluation Metrics

`finger-eval` matches predicted events to ground truth by **temporal IoU** on inclusive frame intervals (default threshold 0.5):

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\max(0,\, \min(e_A, e_B) - \max(s_A, s_B) + 1)}{(e_A - s_A + 1) + (e_B - s_B + 1) - |A \cap B|}
$$

Greedy matching: each GT event is paired with the highest-IoU unmatched prediction of the same class. Reports per-class and overall precision, recall, F1, plus start/end frame timing error.

---

## 11. Complexity Notes

| Stage | Dominant cost |
| --- | --- |
| MediaPipe VIDEO | ~35–80 fps @ 1080p (CPU) |
| Model inference | Chunked ST-GCN + TCN on CPU/GPU |
| Overlay video | Second full video pass + encoding |

**Practical tips:**
- Preprocess once; reuse `.npz` for training experiments.
- Skip `--out-video` for batch runs (JSON only is much faster).
- Use `--include-frames` only when per-frame debugging is needed (large JSON).

---

## 12. File Map

| Module | Role |
| --- | --- |
| `hand_tracking.py` | MediaPipe wrapper |
| `features.py` | Normalization + 12-D features |
| `preprocess.py` | Video → `.npz` |
| `dataset.py` | Chunk dataset + boundary targets |
| `model.py` | ST-GCN + MS-TCN + event post-process |
| `train.py` | Training loop + losses |
| `detect.py` | Video → events JSON |
| `eval_events.py` | Event-level metrics |

