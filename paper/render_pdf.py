"""Markdown -> PDF via headless Chrome.  python paper/render_pdf.py IN.md OUT.pdf  (needs markdown)"""

import os
import subprocess
import sys

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """@page{size:A4;margin:16mm}
body{font-family:-apple-system,'Helvetica Neue',Arial,sans-serif;font-size:10pt;line-height:1.45;color:#1a1a1a}
h1{font-size:17pt;line-height:1.25;margin:0 0 8px}h2{font-size:12.5pt;margin:18px 0 6px;border-bottom:1px solid #ddd;padding-bottom:3px}
h3{font-size:10.5pt;margin:12px 0 4px}table{border-collapse:collapse;width:100%;margin:8px 0;font-size:8.6pt;page-break-inside:avoid}
th,td{border:1px solid #ccc;padding:4px 6px;vertical-align:top;text-align:left}th{background:#f2f2f2}
code{font-family:Menlo,monospace;font-size:8.4pt;background:#f5f5f5;padding:1px 3px;border-radius:3px}
pre{background:#f5f5f5;padding:8px;border-radius:4px}pre code{background:none}hr{border:0;border-top:1px solid #ddd}
img{max-width:100%;display:block;margin:10px auto 2px}"""

src, out = sys.argv[1], os.path.abspath(sys.argv[2])
body = markdown.markdown(open(src).read(), extensions=["tables", "fenced_code"])
html = os.path.join(os.path.dirname(os.path.abspath(src)), ".render.html")   # so figs/ resolves
with open(html, "w") as fh:
    fh.write("<!doctype html><html><head><meta charset='utf-8'><style>%s</style></head><body>%s"
             "</body></html>" % (CSS, body))
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                "--print-to-pdf=" + out, "file://" + html], check=True, capture_output=True)
os.remove(html)
print("->", out)
