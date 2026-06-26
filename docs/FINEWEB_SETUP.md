# FineWeb Setup

Last updated: 2026-05-24

Current state: FineWeb is loaded through the shared corpus utilities, which preserve newline structure and support the probe-ready training flow.

## Dataset path used by the code

- data/fineweb_100mb.txt

This is the path used by corpus_utils.py for the shared FineWeb dataset name.

## Download helpers in this repo

- helpers/download_fineweb.py
- helpers/download_fineweb.ps1

## Verify dataset file

```powershell
Get-Item data\fineweb_100mb.txt
Get-Content data\fineweb_100mb.txt -TotalCount 3
```

## Train with FineWeb

- In auto_train.py, choose FineWeb dataset
- In train.py, FineWeb is attempted first via shared corpus loader, with minimal fallback if missing

## Practical note

Large corpora are often limited for manageable runs (for example around 5000 docs in default flows) to keep runtime and VRAM behavior stable on GT730-class hardware.
