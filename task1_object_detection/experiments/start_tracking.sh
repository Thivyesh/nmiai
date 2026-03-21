#!/bin/bash
# Start MLflow UI and TensorBoard side by side
# MLflow: http://localhost:5000 (experiment comparison, metrics, artifacts)
# TensorBoard: http://localhost:6006 (live training curves)

set -e
cd "$(dirname "$0")/../.."

echo "Starting MLflow UI at http://localhost:5000"
mlflow ui --host 127.0.0.1 --port 5000 &
MLFLOW_PID=$!

echo "Starting TensorBoard at http://localhost:6006"
tensorboard --logdir task1_object_detection/output/models --host 127.0.0.1 --port 6006 &
TB_PID=$!

echo ""
echo "=== Tracking servers running ==="
echo "  MLflow:      http://localhost:5000"
echo "  TensorBoard: http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop both"

trap "kill $MLFLOW_PID $TB_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
