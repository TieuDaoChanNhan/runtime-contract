#!/usr/bin/env bash
set -e

echo "🚀 Provisioning A100-80GB on RunPod..."

# Create the pod
# - Image: Standard PyTorch with CUDA 12.1 (Compatible with FlashAttn 2)
# - GPU: NVIDIA A100 80GB (The paper's target hardware)
# - Disk: 100GB Container / 100GB Volume (Crucial for datasets)
# - Ports: 22 (SSH) and 8000 (vLLM)
POD_INFO=$(runpodctl create pod \
    --name "gpu-training" \
    --imageName "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04" \
    --gpuType "NVIDIA A100 80GB PCIe" \
    --gpuCount 1 \
    --containerDiskSize 100 \
    --volumeSize 100 \
    --ports "22/tcp,8000/tcp" \
    --env "JUPYTER_PASSWORD=changeme" \
    --templateId "runpod-pytorch" \
    --output json)

POD_ID=$(echo "$POD_INFO" | jq -r '.id')

echo "✅ Pod Created! ID: $POD_ID"
echo "⏳ Waiting for SSH to become ready..."

# Wait loop until SSH is accessible (RunPod takes ~30-60s to boot)
while ! runpodctl get pod "$POD_ID" | grep -q "RUNNING"; do
    sleep 5
    echo -n "."
done

echo ""
echo "🎉 Pod is RUNNING."
echo ""
echo "To connect and set up (Copy-paste this):"
echo "---------------------------------------------------"
# Use runpodctl to grab the SSH command, then append your provisioning instruction
SSH_CMD=$(runpodctl ssh "$POD_ID" --print)
echo "$SSH_CMD 'git clone <your-repo-url> repo && bash repo/experiments/cap_sweep/runners/provision.sh'"
echo "---------------------------------------------------"