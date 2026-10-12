"""Build the multi-file ACL paper using an existing pdfLaTeX installation."""
from pathlib import Path
import os
import shutil
import subprocess

PAPER = Path(__file__).resolve().parent
BUILD = PAPER / 'build'
BUILD.mkdir(exist_ok=True)

def executable(name):
    found = shutil.which(name)
    if found:
        return found
    local = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/MiKTeX/miktex/bin/x64' / (name + '.exe')
    if local.is_file():
        return str(local)
    raise FileNotFoundError(f'Install a TeX distribution or add {name} to PATH.')

latex = executable('pdflatex')
bibtex = executable('bibtex')
passes = [
    [latex, '-interaction=nonstopmode', '-halt-on-error', '-output-directory=build', 'main.tex'],
    [bibtex, 'build/main'],
    [latex, '-interaction=nonstopmode', '-halt-on-error', '-output-directory=build', 'main.tex'],
    [latex, '-interaction=nonstopmode', '-halt-on-error', '-output-directory=build', 'main.tex'],
]
for index, command in enumerate(passes, 1):
    result = subprocess.run(command, cwd=PAPER, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = result.stdout.decode('utf-8', errors='replace')
    (BUILD / f'pass_{index}.txt').write_text(output, encoding='utf-8')
    print(f'Pass {index}: {Path(command[0]).name}, exit {result.returncode}', flush=True)
    if result.returncode:
        print(output[-7000:])
        raise SystemExit(result.returncode)
# Line-number references can need one additional pass after appendix edits.
for _ in range(2):
    current_log=(BUILD/'main.log').read_text(encoding='utf-8',errors='replace')
    if 'Line number reference failed' not in current_log:
        break
    result=subprocess.run(passes[-1],cwd=PAPER,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Additional pdfLaTeX pass failed')
    print('Additional line-reference pass: exit 0',flush=True)
shutil.copy2(BUILD/'main.pdf', PAPER/'main.pdf')
log=(BUILD/'main.log').read_text(encoding='utf-8', errors='replace')
for line in log.splitlines():
    if any(term in line for term in ('Output written', 'Overfull', 'undefined', 'Warning:')):
        print(line)
print('PDF:', PAPER/'main.pdf')
