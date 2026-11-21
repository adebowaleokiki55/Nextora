"""Process memory diagnostics (Linux /proc and cgroup v2; empty elsewhere)."""

from __future__ import annotations


def read_process_memory() -> dict:
    result = {
        "rss_mb": None,
        "peak_rss_mb": None,
        "threads": None,
        "cgroup_memory_current_mb": None,
        "cgroup_memory_limit_mb": None,
    }

    try:
        with open("/proc/self/status", "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("VmRSS:"):
                    result["rss_mb"] = round(int(line.split()[1]) / 1024, 2)
                elif line.startswith("VmHWM:"):
                    result["peak_rss_mb"] = round(int(line.split()[1]) / 1024, 2)
                elif line.startswith("Threads:"):
                    result["threads"] = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass

    try:
        with open("/sys/fs/cgroup/memory.current", "r", encoding="utf-8") as file:
            result["cgroup_memory_current_mb"] = round(
                int(file.read().strip()) / (1024 * 1024), 2
            )
    except (OSError, ValueError):
        pass

    try:
        with open("/sys/fs/cgroup/memory.max", "r", encoding="utf-8") as file:
            value = file.read().strip()
            if value != "max":
                result["cgroup_memory_limit_mb"] = round(int(value) / (1024 * 1024), 2)
    except (OSError, ValueError):
        pass

    return result
