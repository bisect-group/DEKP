# emulator_bench (DEKP)

Multi-seed evaluation of DEKP with its original settings on the default EMULaToR splits.
Tasks: `km` (paper) and `kcat`. No tuning. Run commands are in `commands.txt`.

## Pipeline

1. `prepare_splits.py` picks one structure per row and writes manifests under `outputs/baselines/DEKP/<task>/manifests`. Split files are never changed. Order: the best-covering experimental chain that the 3D dataset lists for the row (`pdbs`, `pdb_chain`, `pdb_coverage`), read from `data/intermediate/mmcif/<ID>_updated.cif` and written as a single-chain PDB file under `manifests/_chains`; else AlphaFold, then ESMFold2, by the row's `uniprot_id`. Rows with none of these are dropped and counted.
2. `cache_embeddings.py` (`--dry_run` counts the missing items and exits) builds ProtT5 and SMILES Transformer vectors as the original extraction scripts do (the SMILES Transformer positional encoding depends on the molecule's column in a chunk of 100; the column comes from the row order of `<task>_kinetic_params_1d.parquet`), unpadded token-id and structure-graph caches once per unique entity. Tokens come from the ProtT5 tokenizer and the MoLFormer vocabulary (`bert_vocab.txt`), as in the original. Padding happens at training time, to the longest sequence and SMILES of the run.
3. `train_single_target_tvt.py` trains the original DEKP `MetaDecoder` (`DEKP/pretrain.py`) on the train split, then fits sklearn `ExtraTreesRegressor` on its latent vector. Test metrics come from the Extra Trees.
4. `run_split_benchmarks.py` runs steps 2 and 3 for all seeds and split folders and writes the summary CSVs. `--gpus 0 1 2` gives each training run one GPU through `CUDA_VISIBLE_DEVICES`, and `--runs_per_gpu N` sets the runs per GPU. Without `--gpus`, runs go one at a time on `--device`. `--amp` trains the network in bf16 (the kcat runs use it by the user's decision) (Ampere or newer; fp16 with GradScaler on older GPUs). The default is the original fp32. Extra Trees runs on CPU and ignores `--amp`. With `--amp`, one run needs about 10 GB for the data batches and activations. Use `--runs_per_gpu 1` on 22 GB GPUs.
   Training copies each batch to the GPU on a side stream while the previous batch trains. This does not change the batches or their order. DataLoader workers stay alive between epochs. `--no-persistent_workers` turns this off.

## Original settings

Features `trfm,t5`, hidden 32, 3 layers, kernel 9, dropout 0.5, AdamW lr 1e-3, weight decay 3e-4, MSE loss, batch 256, 200 epochs, last-epoch weights (no early stopping). Seeds 666 777 888.
See `REPORT.md` for sources and differences.

## Data layout

Input: `data/processed/datasets/kinetic_dataset/<task>/<split_group>[/threshold_<t>]/{train,val,test}.parquet` and `<task>_kinetic_params_3d.parquet` for structure lookup. Structure folders: `data/intermediate/{mmcif,alphafold,esmfold2_pdb}`.
Rows without a structure are dropped, as in the original model.

Output: `outputs/baselines/DEKP/<task>/<split_group>/<split_name>/seed_<s>/`, with summaries in `outputs/baselines/DEKP/<task>/`. A split group without threshold folders uses `all` as `<split_name>`.

## Verify equivalence

```
emulator_bench/verify_equivalence.sh          # CPU, fp32, about 20 s
emulator_bench/verify_equivalence.sh --gpu    # also the bf16 tests; needs one CUDA device
emulator_bench/verify_equivalence.sh --negative-control  # the bf16 test must FAIL
```

Expected env: `mldb` (torch, torch_geometric, biotite, scikit-learn, pytest). The script does not create it. It fails if any CPU test is skipped. It needs no private data: it builds a toy cache and synthetic structures.

What `tests/test_original_equivalence.py` compares:
- Training: the loop of `DEKP/pretrain.py` against the wrapper loop (persistent workers, RNG sync, memory-mapped cache, `_prefetch`). Batch hashes, per-step losses and final weights must be identical. The original `pretrain.py` unpacks five batch items where its dataset returns nine, so the test runs its loop statements with the nine items unpacked.
- Persistent workers: without `_sync_rng` the shuffle order changes from epoch 2 on.
- Latents and Extra Trees: `fine_tune.prepare_ds` against `extract_latents`. Predictions must be identical.
- Graphs: `build_graph_from_pdb` against the original `get_graph_fea`, and `extract_chain` against whole-entry PDB files. With local structure data the test uses real files.
- `--gpu`: bf16 autocast with the side-stream copy and `cudnn.benchmark` (as in the training script on CUDA) against fp32 on the same GPU. Dropout is removed from all runs, because dropout masks differ between dtypes. The toy is calm: 96 rows, batch 48, lr 1e-4, 3 epochs, so the bf16 noise floor is small. The noise control is the production run with initial weights perturbed by one rounding of the production dtype (2**-8, bf16). Per step, the gap |production - fp32| must be at most 10 times the floor |perturbed production - production|. The same holds for the final MSE. Step 0 is judged like every other step: its gap must be at most 10 times its floor. Each test prints `max gap`, `max floor` and `ratio`; run with `-s` to see the line. The side-stream batches must equal the plain loader batches. `--negative-control` reruns the bf16 test with the production run's initial weights multiplied by 1.05 (`EQUIV_NEGATIVE_CONTROL=1`). It must fail, and the script exits 0 only then. The test does not cover `--compile_model`: the run records do not show that production runs used it.

Not exact: bf16 training (production setting) differs from fp32 by rounding. With pyg-lib the original `get_graph_fea` returns edges longer than the radius, so the graph test compares with a contiguous copy (see `REPORT.md`). The original `ExtraTreesRegressor` has no seed, so the wrapper seeds it.
