"""
The plotea "hello world": a whole-world map with country outlines.

Run
---
    python examples/01_hello_map.py

Saves ``hello_map.png`` next to this file and prints one log line telling you
which projection and resolution were used.

"""
from pathlib import Path

import plotea

plotea.init_logging()

bm = plotea.BaseMap()
fig, ax = bm.plot()

out = Path(__file__).with_name('hello_map.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'wrote {out}')
