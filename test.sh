# DEPRECATED: use 'python main.py test <config> [--cuda <id>]' instead.
echo "Warning: test.sh is deprecated. Use 'python main.py test <config> [--cuda <id>]' instead." >&2
CUDA=${1:-0}
shift 2>/dev/null; true
CUDA_VISIBLE_DEVICES=$CUDA python tools/train_val.py --config $@ -e