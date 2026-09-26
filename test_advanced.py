"""Quick test of new advanced features on small sample."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'business_entity_resolution', 'src'))

from pipeline import run
import time

t0 = time.time()
run(
    data_dir="dataset",
    out_dir="output_test",
    n_splits=2,
    sample_s1=5000,
    max_df=100,
    min_len=5,
    neg_per_pos_cap=10,
    test_chunk_size=5000,
    use_faiss=False,  # Set True if faiss-cpu installed
    use_cross_encoder=False,  # Set True if transformers installed
    save_model_dir="models_test",
)
print(f"Total time: {time.time() - t0:.1f}s")