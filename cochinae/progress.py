"""Live progress bars for the long-running jobs (video detection).

    python -m cochinae.progress              # watches $MTAH_DATA_DIR/logs/*.log, refreshes every 2 s

Each job writes lines like "video frame 65000  processed 740/2000  3.7 fps  eta 6 min" to its
log; this shows the latest one per log as a bar. A log whose last line starts with "done:" is
shown as finished.
"""

import glob
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

LOGS = os.path.join(DATA_DIR, "logs")
LINE = re.compile(r"processed (\d+)/(\d+)\s+([\d.]+) fps\s+eta (\d+) min")
WIDTH = 32


def bar(frac):
    full = int(round(frac * WIDTH))
    return "█" * full + "░" * (WIDTH - full)


def render():
    rows = []
    for path in sorted(glob.glob(os.path.join(LOGS, "*.log"))):
        name = os.path.basename(path)[:-4]
        with open(path, errors="replace") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        done = any(l.startswith("done:") for l in lines)
        last = next((m for m in map(LINE.search, reversed(lines)) if m), None)
        age = time.time() - os.path.getmtime(path)
        if done:
            rows.append("%-14s %s 100%%  terminado" % (name, bar(1.0)))
        elif last:
            a, b, fps, eta = int(last.group(1)), int(last.group(2)), last.group(3), last.group(4)
            state = "  (sin avance hace %d min)" % (age // 60) if age > 300 else ""
            rows.append("%-14s %s %3d%%  %d/%d  %s fps  faltan ~%s min%s"
                        % (name, bar(a / b), 100 * a // b, a, b, fps, eta, state))
        else:
            rows.append("%-14s %s   ..  arrancando" % (name, bar(0.0)))
    return rows or ["(no hay corridas en %s)" % LOGS]


def main():
    while True:
        rows = render()
        sys.stdout.write("\033[H\033[J" + time.strftime("%H:%M:%S") + "  corridas de Cochinae\n\n"
                         + "\n".join(rows) + "\n\nCtrl-C para salir\n")
        sys.stdout.flush()
        time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
