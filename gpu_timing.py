import pycuda.driver as cuda

class GpuProfiler:
    """
    Zero-allocation CUDA event profiler.
    Pre-allocates events to measure GPU execution time without altering the execution schedule.
    """
    def __init__(self, keys):
        self.events = {key: (cuda.Event(), cuda.Event()) for key in keys}
        self.metrics = {key: 0.0 for key in keys}
        self.active_keys_this_step = set()
        self.accumulation_steps = 0
        self.sync_event = cuda.Event()  # Dedicated event to wait for tracked work

    def zone(self, key):
        return _ProfileZone(self, key)

    def synchronize_and_accumulate(self):
        if not self.active_keys_this_step:
            return

        self.sync_event.record()
        self.sync_event.synchronize()

        for key in self.active_keys_this_step:
            start_evt, end_evt = self.events[key]
            elapsed_sec = start_evt.time_till(end_evt) / 1000.0
            self.metrics[key] += elapsed_sec

        self.active_keys_this_step.clear()
        self.accumulation_steps += 1

    def get_averages_and_reset(self):
        """Returns the average times per step and resets the accumulators."""
        if self.accumulation_steps == 0:
            return {k: 0.0 for k in self.metrics}

        avgs = {k: v / self.accumulation_steps for k, v in self.metrics.items()}
        for k in self.metrics:
            self.metrics[k] = 0.0
        self.accumulation_steps = 0
        return avgs


class _ProfileZone:
    def __init__(self, profiler, key):
        self.profiler = profiler
        self.key = key
        self.start_evt, self.end_evt = profiler.events[key]

    def __enter__(self):
        self.start_evt.record()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_evt.record()
        self.profiler.active_keys_this_step.add(self.key)
