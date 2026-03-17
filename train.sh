# DEPRECATED: use 'python main.py train <config> [--cuda <id>]' instead.
echo "Warning: train.sh is deprecated. Use 'python main.py train <config> [--cuda <id>]' instead." >&2
CUDA=${1:-0}
shift 2>/dev/null; true
CUDA_VISIBLE_DEVICES=$CUDA python tools/train_val.py --config $@
