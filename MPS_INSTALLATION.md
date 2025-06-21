# MPS (Metal Performance Shaders) Installation Guide for Chatterbox TTS Server

This guide provides instructions for running Chatterbox TTS Server with MPS acceleration on Apple Silicon Macs (M1/M2/M3 chips).

## Prerequisites

- macOS 12.3 or later
- Apple Silicon Mac (M1/M2/M3 processor)
- Python 3.10 or later
- Xcode Command Line Tools installed (`xcode-select --install`)

## Installation Steps

1. Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install MPS-compatible dependencies:
```bash
pip install --upgrade pip
pip install -r requirements-mps.txt
```

3. Verify PyTorch MPS support:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

## Configuration

1. In `config.yaml`, set the device to "mps" or "auto":
```yaml
tts_engine:
  device: "mps"  # or "auto" to automatically select between MPS/CPU
```

2. Alternatively, you can specify the device when starting the server:
```bash
python server.py --device mps
```

## Verification

After starting the server, check the logs for confirmation that MPS is being used:
```
INFO: Final device selection: mps
INFO: TTS Model loaded successfully on mps...
```

## Performance Tips

1. For best performance, keep your macOS updated to the latest version
2. Close other GPU-intensive applications while using the TTS server
3. Monitor performance using Activity Monitor's GPU history

## Troubleshooting

**Issue:** "MPS not available" error
- Solution: Ensure you're using macOS 12.3+ and Apple Silicon
- Verify Xcode Command Line Tools are installed

**Issue:** Poor performance
- Solution: Try reducing the chunk size in config.yaml
- Monitor memory usage as MPS shares system RAM

**Issue:** Crashes or instability
- Solution: Try with device set to "cpu" to isolate if issue is MPS-related
- Check for macOS updates
