source "$(dirname "$0")/scripts/common_env.sh"

echo "Using Python: $PYTHON_EXEC"
$PYTHON_EXEC "$PROJECT_ROOT/scripts/train_coke_tracker.py" "$@"
