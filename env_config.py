"""
Centralized environment configuration for NVIDIA GeForce GT 730 GPU runtime.

Bootstraps a coherent CUDA 10.1 + MSVC 14.29 environment by importing the
official Visual Studio vcvars environment into the current Python process.
"""

import os
import subprocess

# ============================================================================
# CUDA & MSVC Paths (Kepler GT 730 Target)
# ============================================================================
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin"
MSVC_TOOLSET_VERSION = "14.29"
VS_BUILDTOOLS_ROOT = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
VCVARSALL_BAT = os.path.join(VS_BUILDTOOLS_ROOT, "VC", "Auxiliary", "Build", "vcvarsall.bat")
DEFAULT_MSVC_142_ROOT = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.29.30133"

_DLL_HANDLES = []
_BOOTSTRAP_FLAG = "LLM_GPU5_VCVARS_BOOTSTRAPPED"


def _prepend_path(env_var: str, path_value: str):
    """Prepend a path once while preserving the rest of the variable."""
    if not path_value or not os.path.exists(path_value):
        return

    current = os.environ.get(env_var, "")
    parts = [part for part in current.split(";") if part]
    normalized = os.path.normcase(path_value)
    parts = [part for part in parts if os.path.normcase(part) != normalized]
    os.environ[env_var] = ";".join([path_value] + parts)


def _load_vcvars_environment():
    """Import the official VS Build Tools 14.29 environment into this process."""
    if os.environ.get(_BOOTSTRAP_FLAG) == "1":
        return
    if not os.path.exists(VCVARSALL_BAT):
        return

    command = f'call "{VCVARSALL_BAT}" x64 -vcvars_ver={MSVC_TOOLSET_VERSION} >nul && set'
    completed = subprocess.run(
        command,
        check=True,
        shell=True,
        capture_output=True,
        text=True,
        encoding="mbcs",
        errors="replace",
    )

    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            os.environ[key] = value

    os.environ[_BOOTSTRAP_FLAG] = "1"


def _msvc_root() -> str:
    root = os.environ.get("VCToolsInstallDir", DEFAULT_MSVC_142_ROOT)
    return root.rstrip("\\/")


def _refresh_tool_paths():
    global MSVC_142_ROOT, MSVC_142_BIN, MSVC_142_INC, MSVC_142_LIB
    MSVC_142_ROOT = _msvc_root()
    MSVC_142_BIN = os.path.join(MSVC_142_ROOT, "bin", "Hostx64", "x64")
    MSVC_142_INC = os.path.join(MSVC_142_ROOT, "include")
    MSVC_142_LIB = os.path.join(MSVC_142_ROOT, "lib", "x64")


def enforce_cuda_isolation():
    """Make CUDA and the supported MSVC toolchain dominate the process environment."""
    _load_vcvars_environment()
    _refresh_tool_paths()

    if hasattr(os, "add_dll_directory") and os.path.exists(CUDA_BIN):
        _DLL_HANDLES.append(os.add_dll_directory(CUDA_BIN))

    _prepend_path("PATH", MSVC_142_BIN)
    _prepend_path("PATH", CUDA_BIN)


# ============================================================================
# Auto-Initialization on Import
# ============================================================================
enforce_cuda_isolation()
