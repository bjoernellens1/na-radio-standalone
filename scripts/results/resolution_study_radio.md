# Resolution Study for radio

| Resolution | Time (ms) | Feature Map | Patch Size |
|---|---|---|---|
| (224, 224) | 352.26 | (14, 14) | (16.0, 16.0) |
| (384, 384) | 203.55 | (24, 24) | (16.0, 16.0) |
| (512, 512) | 395.47 | (32, 32) | (16.0, 16.0) |
| (518, 518) | Error | The input resolution must be a multiple of `self.min_resolution_step`. `self.get_nearest_supported_resolution(<height>, <width>) is provided as a convenience API. Input: torch.Size([518, 518]), Nearest: Resolution(height=512, width=512) | - |
| (640, 640) | 871.27 | (40, 40) | (16.0, 16.0) |
| (1024, 1024) | 2601.02 | (64, 64) | (16.0, 16.0) |
