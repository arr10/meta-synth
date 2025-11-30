# Synthetic Meta-Data
## Overview
This project generates synthetic meta-training tasks for better performance on a main task.

## Usage
Run phase 1:
```
python meta.py
```
Run phase 2:
```
python filter.py
```
Run phase 3:
```
python generate.py
```

## Data
auxillary contains all generated meta-tasks. auxillary_full contains only filtered tasks, with generated input-output pairs.
