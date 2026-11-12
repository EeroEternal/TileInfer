"""Weight handling (route O model supply, mmap/offload).

Empty in M1.  Qwen3.6 loads through vllm-ascend's own loader; nothing here is on the request
path or the startup path yet.
"""

from __future__ import annotations

__all__: list = []
