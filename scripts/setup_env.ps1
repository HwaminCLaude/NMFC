# NMFC 환경 설치 스크립트 (Windows PowerShell 5.1+)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

uv venv --python 3.14 .venv
$py = Join-Path $root ".venv\Scripts\python.exe"

uv pip install --python $py torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $py -r requirements.txt

# GPU 스모크 테스트: is_available()만으로는 부족 - 실제 커널 실행까지 확인
& $py -X utf8 -c "import torch; a=torch.randn(2048,2048,device='cuda'); s=(a@a).float().abs().mean().item(); print('TORCH', torch.__version__); print('GPU', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print('MATMUL_OK', s > 0)"
