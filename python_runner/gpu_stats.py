import pynvml
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="GPU Hardware Statistics Monitor")
    parser.add_argument("--interval", type=float, default=0.5, help="Sampling interval in seconds")
    parser.add_argument("--duration", type=float, default=20.0, help="Total duration to monitor in seconds")
    args = parser.parse_args()

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        print(f"Failed to initialize NVML: {e}")
        return

    device_count = pynvml.nvmlDeviceGetCount()
    if device_count == 0:
        print("No NVIDIA GPUs found.")
        return

    # We assume monitoring of the first GPU (index 0)
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    print(f"Monitoring GPU 0: {name}")
    print(f"{'Time(s)':<8} | {'SM Clock(MHz)':<14} | {'Mem Clock(MHz)':<15} | {'VRAM(MB)':<9} | {'GPU Util(%)':<12} | {'Mem Util(%)':<12}")
    print("-" * 85)

    start_time = time.time()
    while time.time() - start_time < args.duration:
        try:
            # Fetch Clocks
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            
            # Fetch Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used = mem_info.used / (1024 ** 2)
            
            # Fetch Utilization
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util_info.gpu
            mem_util = util_info.memory
            
            elapsed = time.time() - start_time
            print(f"{elapsed:<8.1f} | {sm_clock:<14d} | {mem_clock:<15d} | {vram_used:<9.1f} | {gpu_util:<12d} | {mem_util:<12d}")
            
            time.sleep(args.interval)
        except Exception as e:
            print(f"Error reading from NVML: {e}")
            break

    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
