# train/scheduler.py
import math

class CosineWarmupScheduler:
    def __init__(self, max_lr: float, total_steps: int, warmup_steps: int = 200, min_lr_ratio: float = 0.1):
        self.max_lr = max_lr
        self.total_steps = max(total_steps, warmup_steps + 1)
        self.warmup_steps = warmup_steps
        self.min_lr = max_lr * min_lr_ratio

    def get_lr(self, step: int) -> float:
        if step <= self.warmup_steps:
            return self.max_lr * (step / max(1, self.warmup_steps))
        if step > self.total_steps:
            return self.min_lr
        
        decay_ratio = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
