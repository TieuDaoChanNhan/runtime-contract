# 1. Download runpodctl (Mac/Linux)
wget https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 -O runpodctl
chmod +x runpodctl && sudo mv runpodctl /usr/local/bin/

# 2. Authenticate (Get key from RunPod Settings)
runpodctl config --api-key YOUR_RUNPOD_API_KEY

Detach: Once connected, start your training in a tmux session (installed by provision.sh):

```bash
tmux new -s training
make train-persistent
```