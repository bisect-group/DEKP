# DEKP emulator_bench report

## Method summary

DEKP predicts kcat and Km from three inputs. ProtT5 gives a protein sequence vector. SMILES Transformer gives a substrate vector. A graph transformer (GAT) reads a protein graph built from the 3D structure (CA atoms, radius 10, 20 neighbours).
A neural network (`MetaDecoder`: two CNN sequence encoders, the graph encoder and a residual MLP on the pretrained vectors) is trained with MSE.
Its latent vector (fused features) is then given to an Extra Trees regressor. The Extra Trees is the final predictor (paper Fig. 1, Fig. 3; best of nine ML models).
Paper: Wang et al., Briefings in Bioinformatics 2025, 26(2), bbaf187. Original code: `DEKP/pretrain.py` (network training) and `DEKP/fine_tune.py` (latent extraction plus Extra Trees).
The paper gives no epoch, lr or batch numbers. It reports an 80/10/10 random split averaged over five runs.

## Original default settings

| Setting | Value | Source |
|---|---|---|
| Features | `trfm,t5` (SMILES Transformer 1024, ProtT5 1024) | `fine_tune.py` `--fea` default, `pretrain.py` default adds `pst`; paper p.2 says DEKP uses ProtT5 + SMILES Transformer + graph |
| Hidden / layers / kernel | 32 / 3 / 9 | `pretrain.py` argparse |
| Dropout | 0.5 | `pretrain.py` argparse |
| Optimiser | AdamW, lr 1e-3, weight decay 3e-4 (hard-coded) | `pretrain.py` argparse and optimiser line |
| Loss | MSE | `pretrain.py` |
| Batch size | 256 for training | `pretrain.py` `--batch` default (`fine_tune.py` uses 64, but only for latent extraction) |
| Epochs | 200, no scheduler, no early stopping, last epoch kept | `pretrain.py` `--epoch`, `max_patience = num_epochs` |
| Graph | CA atoms, radius 10, 20 neighbours | `extract_pdb_feature.py` `get_graph_fea` defaults (called in its `__main__` to build `pyg_graph.pkl`) |
| Protein tokens | ProtT5 `T5Tokenizer` (vocab 128, `do_lower_case=False`). `U Z O B *` become `X`, trailing `*` is cut, residues are space-joined | `pretrain.py` `KcatDataset` |
| SMILES tokens | `MolTranBertTokenizer` with MoLFormer `bert_vocab.txt`, applied to the SMILES with its characters space-joined first. This splits `Cl`, `Br` and bracket atoms into single characters | `pretrain.py` `KcatDataset`, `tokenizer.py` |
| Padding length | Longest raw sequence and longest raw SMILES of the whole filtered dataset. They set the token length and the `pos_emb` size of `MetaDecoder` | `pretrain.py` main |
| Training target | `Label` in the original. The wrapper uses `log10_value` (`common.py`) | `pretrain.py` |
| Final predictor | `ExtraTreesRegressor()` sklearn defaults on latent only (`fine_tuned` variant) | `fine_tune.py` `run_ML` |
| Seed | 3407 in code; paper-benchmark seeds 666 777 888 used here | `pretrain.py`; camera_ready.tex L379 and REPRODUCING_BASELINES.md |

Paper and code differences:
- The paper does not state epochs, lr or batch size. I used the released code defaults.
- The paper says features are concatenated and fed to Extra Trees (Fig. 1). The code has three Extra Trees variants: latent only (`fine_tuned`), raw plus latent (`combined`), raw only (`Chosen`). I used latent only. The paper reports ET RMSE 0.73 for the reduced vectors and 0.72 for the non-reduced ones.
- The released flow loads weights pretrained on a larger dataset (`Model/Pretrained_model_199_trfm,t5.pkl`) and does not train in `fine_tune.py`. That dataset overlaps the benchmark test splits. The wrapper trains the network from scratch on each train split instead.
- The original filters (`Label != 1e-8`, no `.` in SMILES) run on the whole dataset before its random split. The wrapper runs them on the train split only.
- Original `pretrain.py` `evaluate` (line ~143) unpacks 5 items from a 9-item batch and crashes. The wrapper follows the evaluation of `fine_tune.py`.
- `ResidualMLP` applies dropout at evaluation time in the original (`F.dropout` without the training flag, `pretrain.py:365-436`). The wrapper keeps this. Latents are stochastic, so the Extra Trees results depend on the order of random calls.
- `fine_tune.py` uses `drop_last=True` on val/test loaders and drops rows. The wrapper evaluates all rows.

## What the wrapper does now

For `--task km|kcat` it runs three steps on every discovered split folder and seed.
1. `prepare_splits.py` resolves one structure per row from the 3D parquet (experimental PDB with at least 90 % identity, else AlphaFold, else ESMFold2). Rows without a structure are dropped. Output: manifests.
2. `cache_embeddings.py` builds ProtT5, SMILES Transformer, unpadded token-id and graph caches once. One cache serves all splits.
3. `train_single_target_tvt.py` pads the token ids to the longest raw sequence and SMILES of the run (train, val and test rows kept after the filters), trains the original `MetaDecoder` with those lengths for 200 epochs, extracts latents for train/val/test, fits Extra Trees, and writes metrics (RMSE, MSE, MAE, R2, Pearson, Spearman). `final_results_test.csv` holds the Extra Trees metrics. `*_nn.csv` hold the metrics of the network output alone.
4. `run_split_benchmarks.py` loops over split folders and seeds and writes the summary CSVs (mean and variance over seeds).

Split folders (from disk, task `km`): `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`, `enzyme_sequence_splits/threshold_0.05`, `substrate_splits/threshold_0.445`. kcat uses `threshold_0.01` and `threshold_0.205`.
Outputs go to `EMULaToR/outputs/baselines/DEKP/<task>/{manifests,embeddings,<split_group>/<split_name>/seed_<s>}` (git-ignored). Nothing is written under `data/`.

## Files removed or changed

Removed: `tune_optuna.py`, `launch_parallel_optuna.py`, `launch_parallel_retrain_from_optuna.py`, `aggregate_dekp_optuna_results.py`, `configs/fine_tune_defaults.json` (defaults are now in argparse; the old JSON had batch 64 and seed 3407), `resolve_experimental_pdbs_for_splits.py` (it rewrote split files in place and is not needed).
Changed:
- `common.py`: repo-relative paths (`task_paths`), the five existing split groups, seeds 666 777 888. Removed the `/home/...` defaults and missing split groups.
- `prepare_splits.py`: `--task`; repo-relative structure folders; ESMFold2 naming; alignment cache moved into the manifests folder; reads `<task>_kinetic_params_3d.parquet`.
- `cache_embeddings.py`: `--task`; always uses manifests; removed full-dataset mode; `--esm_pdb_dir` with ESMFold2 naming; fixed a variable name clash.
- `train_single_target_tvt.py`: explicit paths required; original defaults (batch 256, epochs 200); Extra Trees stage; structure key follows the cache key.
- `run_split_benchmarks.py`: `--task`, `--out_dir`, seeds 666 777 888, original defaults, no JSON override, summaries written to the output folder.
- `feature_pipeline.py`: `.contiguous()` fix. The graph now uses radius 10 and 20 neighbours, as in `get_graph_fea`.
- `bench_tokenizers.py`: loads the ProtT5 tokenizer and the MoLFormer tokenizer. It builds the ids as the original does. The old custom tokenizers are removed. `--protein_max_len`, `--smiles_max_len` and `--graph_neighbors` are removed.
- `bert_vocab.txt`: MoLFormer vocabulary. Source: `https://raw.githubusercontent.com/IBM/molformer/main/training/bert_vocab.txt` (Apache-2.0). sha256: `0190ba2e9cf5ef6a44cefbec1519d085e02d9df2c4949eb391d2fbcadca3e3cc`.
- `README.md`, `commands.txt`: rewritten; no Optuna text.

## Bugs fixed

1. `cache_embeddings.py`: the local name `progress` in the graph-building block shadowed the imported `progress` function (UnboundLocalError on every run). Renamed to `graph_bar`.
2. `feature_pipeline.py`: `radius_graph` on GPU failed with "Inputs must be contiguous"; every structure was skipped. Added `.contiguous()`. Values do not change.
3. `train_single_target_tvt.py`: the training lookup used `uniprot_id` as structure key, but the cache uses the manifest `pdbs` record. Every row was dropped. It now uses the `pdbs` column.
4. ESM folder: the code expected `ESM3-open-small-<id>.pdb` in `data/intermediate/esm`. Disk has `esmfold2_pdb/ESMFold2-<id>.pdb`.
5. Outputs, caches and the alignment cache were written inside the dataset folders. They now go to `outputs/`.
6. Default split groups listed folders that do not exist and the seed default was 3407.

## Deviations from the original protocol

- None for precision by default. Training, inference and caches are fp32 (original behaviour). `--amp` is an opt-in autocast of the network training forward pass (bf16, or fp16 with GradScaler on older GPUs). TF32 stays off. The paper appendix bf16 claim does not override the original code.
- `--cache_dtype` defaults to float32. Caches built earlier with float16 must be rebuilt (`--overwrite_cache`).
- `ExtraTreesRegressor(n_jobs=-1, random_state=seed)`. The original is unseeded. Seeding makes runs reproducible.
- Original preprocessing (`pretrain.py:531-534`: `Label != 1e-8`, no `.` in SMILES) is applied to the TRAIN split only, using `value` for the label filter. The training target is `log10_value`. Val and test rows are not filtered, so every baseline keeps the same test rows (except rows without a structure).
- Structure graphs, token ids and embeddings are cached and read through DataLoader workers. Workers memory-map the cache files and read no row twice. A thread pins the next batch and copies it on a side stream while the current batch trains. Workers persist across epochs. One extra RNG draw before each re-iteration keeps the shuffle order and the global RNG state equal to non-persistent workers. A toy-loader check runs at start-up.
- Per-sample structure is chosen by `prepare_splits.py` (no original equivalent; the original used one fixed structure per UniProt ID).

## Commands

See `emulator_bench/commands.txt`. Short form, from `baselines/DEKP`:

```
for task in km kcat; do python emulator_bench/prepare_splits.py --task $task; done
for task in km kcat; do python emulator_bench/run_split_benchmarks.py --task $task --seeds 666 777 888 --device cuda:0; done
```

Env: `/home/adhil/.conda/envs/mldb/bin/python` (torch, torch_geometric, torch_scatter, transformers, Biopython, sklearn all present). ProtT5 loads from the Hugging Face cache; set `HF_HUB_OFFLINE=1` offline.

## Smoke test

Task km, split `random_splits_grouped_sequence`, 64/16/16 sampled rows (copy in the scratchpad, 3D parquet symlinked), one seed (666), 2 epochs, batch 16, GPU 1 (`CUDA_VISIBLE_DEVICES=1`).
`prepare_splits.py` resolved 36/7/6 rows (rows without a structure dropped). `run_split_benchmarks.py` built the cache (49 graphs), trained, fitted Extra Trees and wrote all output files and four summary CSVs. Output: `/tmp/claude-1005/-home-adhil-github/92fb41ec-b482-467d-940d-08ea9f435509/scratchpad/bench_smoke/DEKP/out`.
The kcat path and the other split folders were not run. They use the same code.

Input-equivalence re-smoke (km, same 36/7/6 rows, seed 666, 2 epochs, batch 16, CPU): the cache was rebuilt with the new tokens and graphs, and training, Extra Trees and all outputs completed. The cache step with the CPU graph path hangs when ProtT5 ran in the same process first (fork after OpenMP threads). Rerun it once so the ProtT5 phase is skipped, or use a CUDA graph device.

## Changes after review

- Summary CSV `dekp_summary_runs.csv`: `train_size`, `val_size`, `test_size` and the ratios now report the rows used (after preprocessing and dropping rows without structure). `orig_train_size`, `orig_val_size`, `orig_test_size` give the split file sizes. `run_summary.csv` also has `orig_*_rows`; `amp_mode` is gone.
- Re-smoke (km, random grouped sequence, 36/7/6 manifest rows, seed 666, 2 epochs, float32 cache, GPU 1): train rows 36 to 34 after preprocessing, Extra Trees fitted, summaries written with used and orig sizes. Output: `bench_smoke/DEKP/out2`.

## Open issues

- Latent-only Extra Trees is the headline result. The paper also shows raw+latent. Pick one if the paper needs the other.
- Training batch: 256 (`pretrain.py`) versus 64 (`fine_tune.py` loaders). I used 256.
- Training from scratch on each train split differs from the released flow (pretraining on a large set). This is required to avoid train/test overlap.
- ProtT5 vectors follow `Seq_to_vec` in `extract_prott5_feature.py`. Sequences over 1000 residues become the first 500 plus the last 500. Only `U Z O B` map to `X`. One sequence per forward pass, fp32 model. The `</s>` position is dropped and the mean is summed in float64. Max abs diff to the original loop (10 sequences, one over 1000, one with U/Z/O/B, GPU fp32 on both sides): 0.0.
- SMILES Transformer vectors follow `extract_smiles_trfm_features.py`: the `split` of the original `utils.py`, 218-token cut (109 + 109), length 220. The original positional encoding slices `pe` by the batch axis, so a molecule's vector depends on its column in the chunk of 100. The wrapper sets the column explicitly. The column of a SMILES is the index of its last row in the full task table `<task>_kinetic_params_1d.parquet` (all rows, stored order, no split filter), modulo 100. This copies the original, where the table is `Dataset.csv` and the dict keeps the last row's vector. The vectors are the same for every split. A SMILES missing from the table gets column 0 and is counted in the log. With the original column, max abs diff to the original code is 0.0 (CPU, 20 and 130 molecules). The same holds for the first 250 rows of the kcat table in order with last-row-wins (191 unique SMILES). A GPU run differs by 1.1e-5. Moving one molecule from column 0 to column 5 changes its vector by up to 0.054.
- Cache folders are versioned (`proteins_v4`, `ligands_v4_<fingerprint>`, `structures_v4`, `cache_version` 4). The ligand folder name holds a hash of the column map, so a reordered table rebuilds the ligand cache. Older caches are ignored.
- The original ran with `transformers==4.43.3`. The mldb env has 5.x, which lower-cases the SMILES tokens in `MolTranBertTokenizer.tokenize`. The wrapper calls the regex split directly, so the ids keep case as in 4.43.3.
- In the mldb env (PyG 2.8, pyg-lib), `get_graph_fea` passes a strided C-alpha view to `radius_graph`. pyg-lib misreads it and returns edges up to 56 A long. The wrapper passes a contiguous copy (`.contiguous()`). Its graph equals `get_graph_fea` with the same fix. With torch-cluster (the original env) the call is not affected.
- Graph is built from the whole PDB file, not one chain, as in the original encoder. `chain_id` is stored in the manifests but not used.
- A full km run has not been timed. 200 epochs with graph batches on 80k rows is slow per seed and split (hours). The runner is sequential.
- Rows without any structure (about 45 % of a random sample) are dropped, so test sets are smaller than the standard splits. Report the kept row counts (`train_rows`, `val_rows`, `test_rows` in `run_summary.csv`).
- `uniprot_time_splits` and other split names are discovered if present; `enzyme_structure_splits` and `conformer_cosine_splits` do not exist on disk.
- The `environment.yml` in the submodule targets an old env; it was not used.
