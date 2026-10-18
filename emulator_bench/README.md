# emulator_bench (DEKP)

Multi-seed evaluation of DEKP with its original settings on the default EMULaToR splits.
Tasks: `km` (paper) and `kcat`. No tuning. Run commands are in `commands.txt`.

## Pipeline

1. `prepare_splits.py` picks one structure per row (experimental, then AlphaFold, then ESMFold2) and writes manifests under `outputs/baselines/DEKP/<task>/manifests`. Split files are never changed.
2. `cache_embeddings.py` builds ProtT5 and SMILES Transformer vectors as the original extraction scripts do (the SMILES Transformer positional encoding depends on the molecule's column in a chunk of 100; the column comes from the row order of `<task>_kinetic_params_1d.parquet`), unpadded token-id and structure-graph caches once per unique entity. Tokens come from the ProtT5 tokenizer and the MoLFormer vocabulary (`bert_vocab.txt`), as in the original. Padding happens at training time, to the longest sequence and SMILES of the run.
3. `train_single_target_tvt.py` trains the original DEKP `MetaDecoder` (`DEKP/pretrain.py`) on the train split, then fits sklearn `ExtraTreesRegressor` on its latent vector. Test metrics come from the Extra Trees.
4. `run_split_benchmarks.py` runs steps 2 and 3 for all seeds and split folders and writes the summary CSVs. `--gpus 0 1 2` gives each training run one GPU through `CUDA_VISIBLE_DEVICES`, and `--runs_per_gpu N` sets the runs per GPU. Without `--gpus`, runs go one at a time on `--device`. `--amp` trains the network in bf16 (Ampere or newer; fp16 with GradScaler on older GPUs). The default is the original fp32. Extra Trees runs on CPU and ignores `--amp`. With `--amp`, one run needs about 10 GB for the data batches and activations. Use `--runs_per_gpu 1` on 22 GB GPUs.
   Training copies each batch to the GPU on a side stream while the previous batch trains. This does not change the batches or their order. DataLoader workers stay alive between epochs. `--no-persistent_workers` turns this off.

## Original settings

Features `trfm,t5`, hidden 32, 3 layers, kernel 9, dropout 0.5, AdamW lr 1e-3, weight decay 3e-4, MSE loss, batch 256, 200 epochs, last-epoch weights (no early stopping). Seeds 666 777 888.
See `REPORT.md` for sources and differences.

## Data layout

Input: `data/processed/datasets/kinetic_dataset/<task>/<split_group>[/threshold_<t>]/{train,val,test}.parquet` and `<task>_kinetic_params_3d.parquet` for structure lookup. Structure folders: `data/intermediate/{processed_exp_pdb,alphafold,esmfold2_pdb}`.
Rows without a structure are dropped, as in the original model.

Output: `outputs/baselines/DEKP/<task>/<split_group>/<split_name>/seed_<s>/`, with summaries in `outputs/baselines/DEKP/<task>/`. A split group without threshold folders uses `all` as `<split_name>`.
