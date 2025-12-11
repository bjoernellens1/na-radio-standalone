# Resolution Study for radio_no_naclip

| Resolution | Time (ms) | Feature Map | Patch Size |
|---|---|---|---|
| (224, 224) | 128.57 | (14, 14) | (16.0, 16.0) |
| (384, 384) | 163.79 | (24, 24) | (16.0, 16.0) |
| (512, 512) | 341.51 | (32, 32) | (16.0, 16.0) |
| (518, 518) | Error | The input resolution must be a multiple of `self.min_resolution_step`. `self.get_nearest_supported_resolution(<height>, <width>) is provided as a convenience API. Input: torch.Size([518, 518]), Nearest: Resolution(height=512, width=512) | - |
| (640, 640) | 1011.67 | (40, 40) | (16.0, 16.0) |
| (1024, 1024) | 5937.36 | (64, 64) | (16.0, 16.0) |
