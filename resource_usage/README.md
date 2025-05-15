# Resource Usage

The `monitor.py` script can monitor a PID and log its CPU and memory usage to a CSV file.
Make sure the following packages are installed:
```bash
pip install psutil
```

## Usage
```bash
python monitor.py <PID>
```
The output will be saved to `pid_{pid}_monitor.csv` in the current directory.

## Proposed Execution Scenario

1. Use screen to run tup-backend in a separate terminal:
```bash
screen -S tup-backend
```
2. Run the tup-backend in the screen session:
```bash
./tup-backend [params]
```
3. Detach from the screen session:
```bash
Ctrl + A, D
```
4. Find the PID of the tup-backend process:
```bash
ps aux | grep tup-backend
```
or
```bash
pgrep -u <user> tup-backend
```
5. Run the monitor script with the PID:
```bash
python monitor.py <PID>
```