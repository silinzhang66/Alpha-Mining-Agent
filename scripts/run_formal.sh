#!/bin/zsh
# Formal experiments (frozen design). Runs that already have results are skipped,
# so the script can be restarted after an interruption.
#   scripts/run_formal.sh deepseek   # Groups A/B/C x seeds 0-4, then S1/S2 (thinking, seeds 0-2)
#   scripts/run_formal.sh qwen       # Groups A/B/C x seeds 0-4
#   scripts/run_formal.sh gp         # GP(200) and GP(1000) x seeds 0-4
cd "${0:A:h}/.."
runs=results/runs
run() {  # $1 = run name, rest = run_mining.py args
  local name=$1; shift
  if [[ -f $runs/$name.meta.json ]]; then echo "skip $name"; return; fi
  echo "$(date +%T) start $name"
  uv run python scripts/run_mining.py "$@" || echo "$(date +%T) FAILED $name"
}
case $1 in
  deepseek|qwen)
    for s in 0 1 2 3 4; do for g in A B C; do
      run ${1}_${g}_s$s llm --model $1 --group $g --seed $s
    done; done
    if [[ $1 == deepseek ]]; then
      for s in 0 1 2; do for g in A C; do
        run deepseek_${g}_think_s$s llm --model deepseek --group $g --seed $s --thinking
      done; done
    fi ;;
  gp)
    for b in 200 1000; do for s in 0 1 2 3 4; do
      run gp${b}_s$s gp --budget $b --seed $s
    done; done ;;
esac
echo "$(date +%T) done $1"
