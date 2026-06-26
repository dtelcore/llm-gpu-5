# Download FineWeb 100MB English Dataset
# Run from project root: .\download_fineweb.ps1

Write-Host ""
Write-Host "Downloading FineWeb 100MB English dataset..."
Write-Host ""

# Run the Python downloader
python download_fineweb.py
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "Download complete!"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Run pipeline: python pipeline.py"
    Write-Host "  2. Select dataset: fineweb_100mb_eng"
    Write-Host "  3. Follow prompts to complete training setup"
}
else {
    Write-Host "Error: Download failed. Check error messages above."
}

exit $exitCode
