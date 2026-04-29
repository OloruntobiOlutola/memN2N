# ============================================================
# Part III — Semi-Supervised MemN2N
# ============================================================
#
# Motivation:
#   The baseline MemN2N is fully unsupervised: it receives no
#   information about which sentences in the story are relevant
#   to answer the question. The original supervised MemNN (Weston
#   et al., 2015) uses supporting-fact labels at every layer,
#   which gives it a large accuracy advantage on hard tasks.
#
#   Here we explore a middle ground: a semi-supervised variant
#   that uses supporting-fact labels for a fraction of training
#   examples (10 %, 25 %, 50 %) to guide the attention at hop 1
#   via an auxiliary KL-divergence loss, while the remaining
#   examples are trained with the standard QA cross-entropy loss
#   only. The architecture is identical to Paul's baseline —
#   only the training loop changes.
#
# Research question:
#   How much does soft attention supervision improve accuracy on
#   tasks where the baseline fails (tasks 2, 3, 15, 16)?
# ============================================================

import os
import glob
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------------------------------------------------------
# Reuse the constants already defined by Paul in the notebook
# (DEVICE, BABI_DIR, SEED, EMBED_DIM, N_HOPS, BATCH_SIZE,
#  N_EPOCHS, LR, ANEAL_STEP, GRAD_CLIP are expected to be in scope)
# -------------------------------------------------------------------

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def _get_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        torch.zeros(1).cuda()
        return torch.device('cuda')
    except Exception:
        return torch.device('cpu')

DEVICE = _get_device()
print(f'Using device: {DEVICE}')
# Download bAbI dataset if not already present
BABI_URL = 'https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz'
BASE_DIR = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
BABI_DIR = os.path.join(BASE_DIR, 'babi_data')

EMBED_DIM   = 20    # paper uses 20 for per-task training
N_HOPS      = 3
BATCH_SIZE  = 32
N_EPOCHS    = 100
LR          = 0.01
ANEAL_STEP = 25    # halve LR every 25 epochs
GRAD_CLIP   = 40.0
N_RUNS      = 3     # paper uses 10; reduce for speed

# ============================================================
# 1. Data loading — with supporting-fact labels
# ============================================================

def parse_babi_with_supports(filepath):
    """
    Parse a bAbI task file into (story, question, answer, supports) tuples.

    The bAbI format encodes supporting facts as tab-separated sentence
    indices on the question line, e.g.:
        "3 Where is John?\\tgarden\\t2"
                                  ^--- 1-based index of the support sentence

    Parameters
    ----------
    filepath : str

    Returns
    -------
    list of (story, question, answer, supports)
        story    : list of token lists  (one per sentence)
        question : list of tokens
        answer   : str
        supports : list of int  (1-based sentence indices)
    """
    data  = []
    story = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            nid, rest = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in rest:
                question, answer, support_str = rest.split('\t')
                question = question.lower().split()
                answer   = answer.lower()
                supports = [int(s) for s in support_str.strip().split()]
                substory = [s for s in story if s]
                data.append((substory, question, answer, supports))
            else:
                story.append(rest.lower().split())
    return data


def build_vocab_ss(data):
    """
    Build vocabulary from (story, question, answer, supports) tuples.
    Index 0 is reserved for padding.
    """
    vocab = set()
    for story, question, answer, _ in data:
        for sentence in story:
            vocab.update(sentence)
        vocab.update(question)
        vocab.add(answer)
    vocab = sorted(vocab)
    word2idx = {w: i + 1 for i, w in enumerate(vocab)}
    idx2word = {i + 1: w for i, w in enumerate(vocab)}
    return word2idx, idx2word


def vectorize_with_supports(data, word2idx,
                              max_story_len, max_sent_len, max_query_len):
    """
    Convert (story, question, answer, supports) tuples into tensors.

    Returns
    -------
    stories      : (N, max_story_len, max_sent_len)   long
    queries      : (N, max_query_len)                 long
    answers      : (N,)                               long
    support_mask : (N, max_story_len)                 float
        Binary mask: 1.0 at the positions of supporting sentences,
        0.0 elsewhere. Used as the soft supervision target for the
        auxiliary attention loss.
    """
    S, Q, A, MASKS = [], [], [], []

    for story, question, answer, supports in data:
        story_len = len(story)

        # --- encode story (same logic as Paul's vectorize_data) ---
        story_vecs = []
        for sent in story[-max_story_len:]:
            sv  = [word2idx.get(w, 0) for w in sent[:max_sent_len]]
            sv += [0] * (max_sent_len - len(sv))
            story_vecs.append(sv)

        # number of zero-sentence rows prepended as left-padding
        pad_len = max_story_len - len(story_vecs)
        while len(story_vecs) < max_story_len:
            story_vecs.insert(0, [0] * max_sent_len)

        # --- build attention supervision mask ---
        mask = [0.0] * max_story_len
        effective_len = min(story_len, max_story_len)
        for sup_idx in supports:
            pos_in_story = sup_idx - 1              # 0-based
            if pos_in_story < effective_len:
                tensor_pos = pad_len + pos_in_story # position after padding
                if tensor_pos < max_story_len:
                    mask[tensor_pos] = 1.0

        # --- encode query and answer ---
        qv  = [word2idx.get(w, 0) for w in question[:max_query_len]]
        qv += [0] * (max_query_len - len(qv))
        av  = word2idx.get(answer, 0)

        S.append(story_vecs)
        Q.append(qv)
        A.append(av)
        MASKS.append(mask)

    return (
        torch.tensor(S,     dtype=torch.long),
        torch.tensor(Q,     dtype=torch.long),
        torch.tensor(A,     dtype=torch.long),
        torch.tensor(MASKS, dtype=torch.float),
    )


# ============================================================
# 2. Model — identical to Paul's MemN2N, extended with
#    return_attention flag to expose per-hop attention weights
# ============================================================

def position_encoding_ss(max_sentence_len, embedding_dim):
    """
    Position encoding (PE) from Section 4.1 of the paper:
        l_kj = (1 - j/J) - (k/d)(1 - 2j/J)   (1-indexed j and k)

    Returns
    -------
    torch.Tensor of shape (max_sentence_len, embedding_dim)
    """
    J = max_sentence_len
    d = embedding_dim
    j = torch.arange(1, J + 1, dtype=torch.float).unsqueeze(1)  # (J, 1)
    k = torch.arange(1, d + 1, dtype=torch.float).unsqueeze(0)  # (1, d)
    L = (1 - j / J) - (k / d) * (1 - 2 * j / J)                # (J, d)
    return L



class MemN2N_SS(nn.Module):
    """
    End-to-End Memory Network — Semi-Supervised variant.

    Architecture is identical to Paul's MemN2N. The only addition is the
    `return_attention` flag in forward(), which exposes the per-hop
    attention weight tensors needed to compute the auxiliary loss.

    Parameters
    ----------
    vocab_size    : int
    embed_dim     : int   — embedding dimension d
    max_sent_len  : int   — maximum words per sentence (for PE)
    max_story_len : int   — maximum sentences in memory
    n_hops        : int   — number of memory hops K (default 3)
    """

    def __init__(self, vocab_size, embed_dim,
                 max_sent_len, max_story_len, max_query_len, n_hops=3):
        super().__init__()
        self.n_hops       = n_hops
        self.embed_dim    = embed_dim
        self.max_sent_len = max_sent_len

        # Adjacent weight tying: A^{k+1} = C^k
        # embeddings[k]   = A^k  (input memory embedding at hop k)
        # embeddings[k+1] = C^k  (output memory embedding at hop k)
        # embeddings[0]   = B    (query embedding)
        # embeddings[K]   = W^T  (final prediction weight)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            for _ in range(n_hops + 1)
        ])
        self.query_emb = self.embeddings[0]
        self.W         = self.embeddings[n_hops]

        # Position encoding — fixed, not learned
        pe_sent  = position_encoding_ss(max_sent_len,  embed_dim)
        pe_query = position_encoding_ss(max_query_len, embed_dim)
        self.register_buffer('pe_sent',  pe_sent)   # (max_sent_len,  d)
        self.register_buffer('pe_query', pe_query)  # (max_query_len, d)

        # Temporal encoding — one matrix per hop, learned (Section 4.1)
        self.TA = nn.ParameterList([
            nn.Parameter(torch.randn(max_story_len, embed_dim) * 0.1)
            for _ in range(n_hops)
        ])
        self.TC = nn.ParameterList([
            nn.Parameter(torch.randn(max_story_len, embed_dim) * 0.1)
            for _ in range(n_hops)
        ])

        self._init_weights()

    def _init_weights(self):
        """Gaussian initialisation σ=0.1 as in Section 4.2."""
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0, std=0.1)
            if emb.padding_idx is not None:
                emb.weight.data[emb.padding_idx].fill_(0)

    def embed_sentences(self, x, emb_matrix, temporal_enc):
        """
        Embed a batch of stories with position and temporal encoding.

        Parameters
        ----------
        x            : (batch, n_sentences, sentence_len)   long
        emb_matrix   : nn.Embedding  — A^k or C^k
        temporal_enc : (n_sentences, embed_dim)  — TA[k] or TC[k]

        Returns
        -------
        (batch, n_sentences, embed_dim)
        """
        word_embs = emb_matrix(x)                          # (B, M, J, d)
        pe        = self.pe_sent.unsqueeze(0).unsqueeze(0)  # (1, 1, J, d)
        word_embs = word_embs * pe
        m         = word_embs.sum(dim=2)
        m         = m + temporal_enc.unsqueeze(0)
        return m

    def forward(self, story, query, return_attention=False):
        """
        Parameters
        ----------
        story            : (batch, n_sentences, sentence_len)
        query            : (batch, query_len)
        return_attention : bool
            If True, also return the list of per-hop attention tensors.

        Returns
        -------
        logits : (batch, vocab_size)
        attentions : list of K tensors of shape (batch, n_sentences)
            Only returned when return_attention=True.
        """
        # Encode query → initial hidden state u^1
        q_len = query.size(1)
        pe_q = self.pe_query.unsqueeze(0)               # (1, max_query_len, d)
        u    = (self.query_emb(query) * pe_q).sum(dim=1) # (batch, d)

        attentions = []

        for hop in range(self.n_hops):
            A_k = self.embeddings[hop]
            C_k = self.embeddings[hop + 1]

            m = self.embed_sentences(story, A_k, self.TA[hop])  # (B, M, d)
            c = self.embed_sentences(story, C_k, self.TC[hop])  # (B, M, d)

            # Attention weights — equation (1)
            p = F.softmax(
                torch.bmm(m, u.unsqueeze(2)).squeeze(2),    # (B, M)
                dim=1
            )
            attentions.append(p)

            # Output vector — equation (2)
            o = (p.unsqueeze(2) * c).sum(dim=1)             # (B, d)

            # Hidden state update — equation (4)
            u = u + o

        logits = F.linear(u, self.W.weight)                 # (B, vocab_size)

        if return_attention:
            return logits, attentions
        return logits


# ============================================================
# 3. Semi-supervised training loop
# ============================================================

def train_semisupervised_memn2n(
        train_S, train_Q, train_A, train_masks,
        test_S,  test_Q,  test_A,
        vocab_size, max_sent_len, max_story_len, max_query_len,
        labeled_ratio=0.1,
        lambda_aux=0.1,
        embed_dim=20,
        n_hops=3,
        n_epochs=100,
        batch_size=32,
        lr=0.01,
        anneal_step=25,
        grad_clip=40.0,
        device=None):
    """
    Train MemN2N_SS with a mixed supervised / unsupervised objective.

    Loss = CrossEntropy(QA)  +  lambda_aux * KLDiv(attention_hop1, support_mask)

    The KL term is computed only for the `labeled_ratio` fraction of
    training examples that have support-fact annotations. All examples
    contribute to the QA cross-entropy loss.

    Parameters
    ----------
    train_S, train_Q, train_A : tensors from vectorize_with_supports()
    train_masks               : (N, max_story_len) float — support mask
    test_S, test_Q, test_A   : standard test tensors (no masks needed)
    vocab_size                : int
    max_sent_len              : int
    max_story_len             : int
    labeled_ratio             : float in [0, 1]
        Fraction of training examples treated as labeled.
        0.0 → purely unsupervised baseline
        1.0 → fully supervised (upper bound)
    lambda_aux                : float
        Weight of the auxiliary attention loss.
    embed_dim, n_hops, ...    : standard hyperparameters

    Returns
    -------
    model      : trained MemN2N_SS
    best_state : state_dict with the lowest training loss
    history    : dict with 'train_loss' and 'test_acc' lists
    """
    if device is None:
        device = torch.device('cpu')

    model     = MemN2N_SS(vocab_size, embed_dim,
                   max_sent_len, max_story_len, max_query_len, n_hops).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_train   = train_S.size(0)
    n_labeled = int(n_train * labeled_ratio)

    print(f"  Total train: {n_train} | "
          f"Labeled: {n_labeled} | "
          f"Unlabeled: {n_train - n_labeled}")

    best_train_loss = float('inf')
    best_state      = None
    history         = {'train_loss': [], 'test_acc': []}

    for epoch in tqdm(range(1, n_epochs + 1), desc="Training", leave=False):
        model.train()

        # Shuffle all training data together
        perm       = torch.randperm(n_train)
        sS         = train_S[perm]
        sQ         = train_Q[perm]
        sA         = train_A[perm]
        sM         = train_masks[perm]
        # Boolean mask: True for examples that have support annotations
        is_labeled = perm < n_labeled

        total_loss = 0.0

        for i in range(0, n_train, batch_size):
            bS  = sS[i:i + batch_size].to(device)
            bQ  = sQ[i:i + batch_size].to(device)
            bA  = sA[i:i + batch_size].to(device)
            bM  = sM[i:i + batch_size].to(device)
            lab = is_labeled[i:i + batch_size]  # bool tensor, CPU

            optimizer.zero_grad()

            # ── QA loss (all examples) ──────────────────────────────
            logits, attentions = model(bS, bQ, return_attention=True)
            loss_qa = criterion(logits, bA)

            # ── Auxiliary attention loss (labeled examples only) ────
            loss_aux = torch.tensor(0.0, device=device)

            if lab.any():
                # Supervise only hop 1: this is where the model should
                # first identify the relevant sentence(s).
                attn_hop1 = attentions[0]           # (batch, n_sents)
                attn_lab  = attn_hop1[lab]           # (n_lab, n_sents)
                mask_lab  = bM[lab]                  # (n_lab, n_sents)

                # Normalise the binary mask into a probability distribution
                # (handles multi-support tasks like task 2 and 3)
                mask_norm = mask_lab / (
                    mask_lab.sum(dim=1, keepdim=True).clamp(min=1e-8)
                )

                # KL divergence: KL(target || predicted)
                # Encourages the learned attention to match the target
                # distribution without forcing a hard argmax.
                loss_aux = F.kl_div(
                    input  = attn_lab.clamp(min=1e-8).log(),  # log q
                    target = mask_norm,                        # p
                    reduction='batchmean'
                )

            loss = loss_qa + lambda_aux * loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * bS.size(0)

        # Learning rate annealing ÷2 every anneal_step epochs (Section 4.2)
        if epoch % anneal_step == 0:
            for pg in optimizer.param_groups:
                pg['lr'] /= 2

        avg_loss = total_loss / n_train
        history['train_loss'].append(avg_loss)

        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(test_S.to(device), test_Q.to(device))
                acc    = (logits.argmax(1).cpu() == test_A).float().mean().item()
            history['test_acc'].append(acc * 100)
            tqdm.write(f"  Epoch {epoch:3d} | "
                       f"Loss: {avg_loss:.4f} | "
                       f"Test Acc: {acc * 100:.1f}%")

    return model, best_state, history


# ============================================================
# 4. Full experiment across tasks and labeled ratios
# ============================================================

TASK_FILES_SS = {
    2:  'two-supporting-facts',
    3:  'three-supporting-facts',
    15: 'basic-deduction',
    16: 'basic-induction',
}

# Paper results for reference (Table 1, PE+LS joint, 1k training)
PAPER_RESULTS_SS = {2: 14.0, 3: 33.1, 15: 0.0, 16: 3.5}

LABELED_RATIOS = [0.0, 0.1, 0.25, 0.5]
LAMBDA_AUX     = 0.1
N_RUNS_SS      = 3   # number of random restarts (paper uses 10)

base_path_ss = os.path.join(BABI_DIR, 'tasks_1-20_v1-2', 'en-10k')

# results[task_id][labeled_ratio] = mean error rate (%)
results_ss = {tid: {} for tid in TASK_FILES_SS}

for task_id, task_name in TASK_FILES_SS.items():
    print(f"\n{'=' * 60}")
    print(f"Task {task_id}: {task_name}")
    print('=' * 60)

    train_file_ss = glob.glob(
        os.path.join(base_path_ss, f'qa{task_id}_*_train.txt'))[0]
    test_file_ss  = glob.glob(
        os.path.join(base_path_ss, f'qa{task_id}_*_test.txt'))[0]

    # Parse with support labels
    train_data_ss = parse_babi_with_supports(train_file_ss)
    test_data_ss  = parse_babi_with_supports(test_file_ss)
    all_data_ss   = train_data_ss + test_data_ss

    # Compute max lengths from the data
    max_story_ss = min(max(len(d[0]) for d in all_data_ss), 50)
    max_sent_ss  = max(len(s) for d in all_data_ss for s in d[0])
    max_query_ss = max(len(d[1]) for d in all_data_ss)

    word2idx_ss, idx2word_ss = build_vocab_ss(all_data_ss)
    vocab_size_ss = len(word2idx_ss) + 1

    train_S_ss, train_Q_ss, train_A_ss, train_masks_ss = \
        vectorize_with_supports(
            train_data_ss, word2idx_ss,
            max_story_ss, max_sent_ss, max_query_ss)

    test_S_ss, test_Q_ss, test_A_ss, _ = \
        vectorize_with_supports(
            test_data_ss, word2idx_ss,
            max_story_ss, max_sent_ss, max_query_ss)

    for ratio in LABELED_RATIOS:
        label = f"{int(ratio * 100)}% labeled"
        print(f"\n  Condition: {label}")

        run_errors = []
        for run in range(N_RUNS_SS):
            torch.manual_seed(run * 100 + task_id)
            model_ss, state_ss, _ = train_semisupervised_memn2n(
                train_S_ss, train_Q_ss, train_A_ss, train_masks_ss,
                test_S_ss,  test_Q_ss,  test_A_ss,
                vocab_size  = vocab_size_ss,
                max_sent_len  = max_sent_ss,
                max_story_len = max_story_ss,
                max_query_len = max_query_ss,
                labeled_ratio = ratio,
                lambda_aux    = LAMBDA_AUX,
                embed_dim     = EMBED_DIM,
                n_hops        = N_HOPS,
                n_epochs      = N_EPOCHS,
                batch_size    = BATCH_SIZE,
                lr            = LR,
                anneal_step   = ANEAL_STEP,
                grad_clip     = GRAD_CLIP,
                device        = DEVICE,
            )
            # Pick best run by training loss
            model_ss.load_state_dict(state_ss)
            model_ss.eval()
            with torch.no_grad():
                logits = model_ss(test_S_ss.to(DEVICE), test_Q_ss.to(DEVICE))
                acc    = (logits.argmax(1).cpu() == test_A_ss).float().mean().item()
            err = (1 - acc) * 100
            run_errors.append(err)

        mean_err = np.mean(run_errors)
        results_ss[task_id][ratio] = mean_err
        print(f"  → Mean error over {N_RUNS_SS} runs: {mean_err:.1f}%")


# ============================================================
# 5. Results table
# ============================================================

print("\n\nSemi-Supervised Results — Error Rate (%)")
print("=" * 72)

header = f"{'Task':<6} {'Name':<28}"
for ratio in LABELED_RATIOS:
    header += f"  {int(ratio*100)}%lab"
header += "  Paper"
print(header)
print("-" * 72)

for task_id, task_name in TASK_FILES_SS.items():
    row = f"  {task_id:<4} {task_name:<28}"
    for ratio in LABELED_RATIOS:
        row += f"  {results_ss[task_id][ratio]:>6.1f}"
    row += f"  {PAPER_RESULTS_SS[task_id]:>6.1f}"
    print(row)

print("-" * 72)
print("Columns: error rate (%) — lower is better.")
print("'0% labeled' = unsupervised baseline (same as Paul's model).")
print("'Paper' = MemN2N PE+LS joint 1k (Table 1).")


# ============================================================
# 6. Plot: error rate vs labeled ratio per task
# ============================================================

fig, axes = plt.subplots(1, len(TASK_FILES_SS),
                          figsize=(4 * len(TASK_FILES_SS), 4),
                          sharey=False)

ratio_labels = [f"{int(r * 100)}%" for r in LABELED_RATIOS]

for ax, (task_id, task_name) in zip(axes, TASK_FILES_SS.items()):
    errors = [results_ss[task_id][r] for r in LABELED_RATIOS]
    paper  = PAPER_RESULTS_SS[task_id]

    ax.plot(ratio_labels, errors, 'o-', color='steelblue',
            linewidth=2, markersize=7, label='Semi-supervised')
    ax.axhline(paper, color='tomato', linestyle='--',
               linewidth=1.5, label=f'Paper ({paper:.1f}%)')
    ax.set_title(f'Task {task_id}\n{task_name}', fontsize=9)
    ax.set_xlabel('Labeled ratio')
    ax.set_ylabel('Error rate (%)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle(
    'Semi-Supervised MemN2N — Error rate vs fraction of labeled examples\n'
    'Auxiliary KL loss on hop-1 attention (λ=0.1)',
    fontsize=11, y=1.02
)
plt.tight_layout()
plt.savefig('semisupervised_results.png', dpi=130, bbox_inches='tight')
plt.show()
print("Figure saved to semisupervised_results.png")


# ============================================================
# 7. Attention visualisation — supervised vs unsupervised
#    (qualitative analysis for the report)
# ============================================================

def visualize_attention_comparison(
        example_idx, test_data, test_S, test_Q, test_A,
        model_unsup, model_sup, idx2word,
        max_story_len, n_hops, device):
    """
    Side-by-side attention heatmap for the unsupervised baseline
    and the semi-supervised model on the same example.
    """
    story_raw, query_raw, answer_raw, supports = test_data[example_idx]

    s = test_S[example_idx].unsqueeze(0).to(device)
    q = test_Q[example_idx].unsqueeze(0).to(device)

    def get_attn(model):
        model.eval()
        with torch.no_grad():
            logits, attn = model(s, q, return_attention=True)
        pred = idx2word.get(logits.argmax(1).item(), '?')
        return [a.squeeze(0).cpu().numpy() for a in attn], pred

    attn_unsup, pred_unsup = get_attn(model_unsup)
    attn_sup,   pred_sup   = get_attn(model_sup)

    n_sents = len(story_raw)
    offset  = max_story_len - n_sents
    labels  = [' '.join(s) for s in story_raw]
    q_str   = ' '.join(query_raw)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(3, n_sents * 0.45)),
                              sharey=True)

    for ax, attn_list, title, pred in [
        (axes[0], attn_unsup, 'Unsupervised (0% labeled)', pred_unsup),
        (axes[1], attn_sup,   'Semi-supervised (50% labeled)', pred_sup),
    ]:
        mat = np.stack([a[offset:] for a in attn_list], axis=1)
        im  = ax.imshow(mat, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(n_hops))
        ax.set_xticklabels([f'Hop {k+1}' for k in range(n_hops)])
        ax.set_yticks(range(n_sents))
        ax.set_yticklabels(labels, fontsize=7)
        # Mark true supporting sentences
        for sup_idx in supports:
            row = sup_idx - 1
            if 0 <= row < n_sents:
                ax.get_yticklabels()[row].set_color('green')
                ax.get_yticklabels()[row].set_fontweight('bold')
        plt.colorbar(im, ax=ax)
        ax.set_title(
            f'{title}\nPredicted: {pred}  |  True: {answer_raw}',
            fontsize=9
        )

    fig.suptitle(f'Q: {q_str}', fontsize=10)
    plt.tight_layout()
    plt.show()


# --- Run the visualisation on 3 examples from task 2 ---
# (requires model_unsup and model_sup to have been trained above)
#
# To use: re-train with labeled_ratio=0.0 and labeled_ratio=0.5
# on task 2, then call:
#
# for i in range(3):
#     visualize_attention_comparison(
#         i, test_data_ss, test_S_ss, test_Q_ss, test_A_ss,
#         model_unsup, model_sup, idx2word_ss,
#         max_story_ss, N_HOPS, DEVICE
#     )
