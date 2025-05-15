import psutil
import time
import argparse
import sys
import csv

def monitor_pid(pid, interval=1):
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"No process with PID {pid} found.")
        sys.exit(1)

    print(f"Monitoring PID {pid} - '{proc.name()}' (press Ctrl+C to stop)\n")
    print(f"{'Time':<20} {'CPU (%)':<10} {'Memory (MB)':<15}")

    # Open a CSV file to write the data
    with open(f"pid_{pid}_monitor.csv", mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header row
        csv_writer.writerow(["Time", "CPU (%)", "Memory (MB)"])

        try:
            while True:
                cpu = proc.cpu_percent(interval=interval)
                mem = proc.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"{timestamp:<20} {cpu:<10.2f} {mem:<15.2f}")
                # Write the data row to the CSV
                csv_writer.writerow([timestamp, cpu, mem])
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
        except psutil.NoSuchProcess:
            print(f"Process with PID {pid} has terminated.") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor CPU and memory usage of a PID.")
    parser.add_argument("pid", type=int, help="Process ID to monitor.")
    parser.add_argument("--interval", type=float, default=1, help="Polling interval in seconds (default: 1)")
    
    args = parser.parse_args()
    monitor_pid(args.pid, args.interval)
