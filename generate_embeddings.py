#!/usr/bin/env python3
"""Stream E5-large-v2 embeddings to disk (RTX 5080, fp16, batch 512)."""
import sys, os, gc, time, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

# Config
MODEL_NAME = "intfloat/e5-large-v2"
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = True
CHECKPOINT_EVERY = 500_000
OUT_DIR = Path("embeddings")
OUT_DIR.mkdir(exist_ok=True)

# Load model
print(f"Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
if FP16:
    model.half()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_texts(texts, prefix=""):
    texts = [prefix + t for t in texts]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model_output = model(**encoded)
    embeddings = mean_pooling(model_output, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy().astype(np.float16)  # fp16 save

def stream_embeddings(texts, entity_ids, out_name, prefix=""):
    """Stream embeddings to memmap + save entity_ids."""
    n = len(texts)
    dim = 1024
    out_path = OUT_DIR / f"{out_name}.npy"
    id_path = OUT_DIR / f"{out_name}_ids.npy"
    
    # Create memmap
    fp = np.memmap(out_path, dtype=np.float16, mode="w+", shape=(n, 1024))
    np.save(id_path, np.array(entity_ids, dtype=object))
    
    written = 0
    for i in range(0, n, BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_ids = entity_ids[i:i+BATCH_SIZE]
        emb = embed_texts(batch_texts, prefix="passage: ")
        fp[written:written+len(batch_texts)] = emb
        written += len(batch_texts)
        if written % CHECKPOINT_EVERY == 0:
            fp.flush()
            print(f"  {out_name}: {written}/{n} ({written/n*100:.1f}%)")
    fp.flush()
    print(f"  {out_name}: DONE {written}/{n}")
    return out_path, id_path

def load_column(path, col):
    import pandas as pd
    df = pd.read_csv(path, sep="\t", usecols=["entity_id", col])
    return df["entity_id"].tolist(), df[col].fillna("").astype(str).tolist()

if __name__ == "__main__":
    DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../dataset")
    
    print("Loading data...")
    # Train
    s1_tr_ids, s1_tr_names = load_column(DATA/"train/train_source1.tsv", "business_name")
    _, s1_tr_addrs = load_column(DATA/"train/train_source1.tsv", "business_address")
    s2_tr_ids, s2_tr_names = load_column(DATA/"train/train_source2.tsv", "business_name")
    _, s2_tr_addrs = load_column(DATA/"train/train_source2.tsv", "business_address")
    s3_tr_ids, s3_tr_names = load_column(DATA/"train/train_source3.tsv", "business_name")
    _, s3_tr_addrs = load_column(DATA/"train/train_source3.tsv", "business_address")
    
    # Test
    s1_te_ids, s1_te_names = load_column(DATA/"test/test_source1.tsv", "business_name")
    _, s1_te_addrs = load_column(DATA/"test/test_source1.tsv", "business_address")
    s2_te_ids, s2_te_names = load_column(DATA/"test/test_source2.tsv", "business_name")
    _, s2_te_addrs = load_column(DATA/"test/test_source2.tsv", "business_address")
    s3_te_ids, s3_te_names = load_column(DATA/"test/test_source3.tsv", "business_name")
    _, s3_te_addrs = load_column(DATA/"test/test_source3.tsv", "business_address")
    
    print("Generating embeddings...")
    t0 = time.time()
    
    # Train names
    all_tr_names = s1_tr_names + s2_tr_names + s3_tr_names
    all_tr_name_ids = s1_tr_ids + s2_tr_ids + s3_tr_ids
    stream_embeddings(all_tr_names, all_tr_name_ids, "train_names", prefix="passage: ")
    
    # Train addresses
    all_tr_addrs = s1_tr_addrs + s2_tr_addrs + s3_tr_addrs
    all_tr_addr_ids = s1_tr_ids + s2_tr_ids + s3_tr_ids
    stream_embeddings(all_tr_addrs, all_tr_addr_ids, "train_addrs", prefix="passage: ")
    
    # Test names
    all_te_names = s1_te_names + s2_te_names + s3_te_names
    all_te_name_ids = s1_te_ids + s2_te_ids + s3_te_ids
    stream_embeddings(all_te_names, all_te_name_ids, "test_names", prefix="passage: ")
    
    # Test addresses
    all_te_addrs = s1_te_addrs + s2_te_addrs + s3_te_addrs
    all_te_addr_ids = s1_te_ids + s2_te_ids + s3_te_ids
    stream_embeddings(all_te_addrs, all_te_addr_ids, "test_addrs", prefix="passage: ")
    
    print(f"TOTAL TIME: {time.time()-t0:.0f}s")
    print("Embeddings saved to embeddings/")