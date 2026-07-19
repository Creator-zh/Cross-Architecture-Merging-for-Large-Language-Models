# Thai / Malay overnight pipeline

## Attach tmux
```bash
tmux attach -t dfop
# window: thai-malay-overnight
```

## Quick status
```bash
cd ~/projects/Cross-Architecture-Merging-for-Large-Language-Models
tail -n 50 evaluation_results/dfop_thai_malay_pipeline/overnight.log
ls evaluation_results/dfop_thai_malay_pipeline/*.done
cat evaluation_results/dfop_thai_malay_pipeline/summary.txt
```

## Stages (markers in `*.done`)
1. `pre_thai_target` / `pre_thai_dfop_full`
2. `pre_malay_target` / `pre_malay_dfop_full`
3. `sft_thai_malay`
4. `post_thai_dfop_sft` / `post_malay_dfop_sft`
5. `pipeline_all` + `summary.txt`

## Resume if interrupted
```bash
bash scripts/run_dfop_thai_malay_overnight.sh
# skips completed *.done stages
```

GPU: 6 (only free card at launch). Offline HF caches enabled.
