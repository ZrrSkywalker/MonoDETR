CUDA=${1:-0}
shift 2>/dev/null; true
CUDA_VISIBLE_DEVICES=$CUDA python tools/train_val.py --config $@
