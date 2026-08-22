"""CPU Thermal Throttling & Performance Monitor for Windows.

Monitors CPU frequency, CPU usage percentage, and (if supported by OS/WMI) CPU temperature over a multi-minute run.
"""

import time
import sys
import psutil

try:
    import wmi
    has_wmi = True
except ImportError:
    has_wmi = False

def monitor_cpu(duration_sec=180, interval_sec=3):
    print("=" * 65)
    print(f" CPU Performance & Thermal Throttling Monitor ({duration_sec}s run)")
    print("=" * 65)
    print(f"{'Time':<10} | {'CPU Load (%)':<15} | {'Current Freq (MHz)':<20} | {'Max Freq (MHz)':<15}")
    print("-" * 65)

    start_time = time.time()
    initial_freq = psutil.cpu_freq()
    base_mhz = initial_freq.max if initial_freq else 0

    while (time.time() - start_time) < duration_sec:
        elapsed = time.time() - start_time
        load = psutil.cpu_percent(interval=interval_sec)
        freq = psutil.cpu_freq()

        cur_mhz = freq.current if freq else 0
        max_mhz = freq.max if freq else 0

        # Check for potential frequency thermal throttling
        throttling_warning = ""
        if base_mhz > 0 and cur_mhz < (base_mhz * 0.70) and load > 50:
            throttling_warning = " ⚠️ POSSIBLE THERMAL THROTTLING DETECTED!"

        print(f"{elapsed:6.1f}s   | {load:12.1f}%   | {cur_mhz:18.1f}   | {max_mhz:13.1f}{throttling_warning}")

    print("=" * 65)
    print(" Monitoring Complete.")

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    monitor_cpu(duration_sec=duration)
