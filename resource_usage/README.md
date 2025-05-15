# Resource Usage

The `monitor.py` script can monitor a PID and log its CPU and memory usage to a CSV file.
Make sure following packages are installed:
```bash
pip install psutil
```

## Usage
```bash
python monitor.py <PID>
```
The output will be saved to `pid_{pid}_monitor.csv` in the current directory.
