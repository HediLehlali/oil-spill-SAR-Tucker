$ErrorActionPreference = "Continue"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"
Set-Location "C:\Users\hedil\Desktop\ELAB4SPACE\Oil Spills"
$models = @("segformer_b2", "upernet_swin_t", "swin_unetr", "transunet", "mask2former_swin_t")
foreach ($m in $models) {
    $log = "outputs\Seg5_Runs\stdout_$m.txt"
    & .\venv\Scripts\python.exe -m seg_bench.run --model $m --epochs 10 --batch-size 8 --workers 4 *> $log
    "$(Get-Date -Format s) finished $m (exit $LASTEXITCODE)" | Add-Content outputs\Seg5_Runs\queue_log.txt
}
