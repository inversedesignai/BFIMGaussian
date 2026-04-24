import re

with open(r'f:\My Drive\BFIMGaussian\doc\paper\paper.tex') as f:
    tex = f.read()

cited = set()
for m in re.finditer(r'\\cite\{([^}]+)\}', tex):
    for key in m.group(1).split(','):
        cited.add(key.strip())

bib_keys = set()
with open(r'f:\My Drive\BFIMGaussian\doc\paper\paper.bib') as f:
    for line in f:
        m = re.match(r'@\w+\{([^,]+),', line)
        if m:
            bib_keys.add(m.group(1).strip())

missing = sorted(cited - bib_keys)
unused = sorted(bib_keys - cited)
print(f'cited: {len(cited)}, bib: {len(bib_keys)}')
print(f'CITED BUT NOT IN BIB ({len(missing)}):')
for k in missing: print(' ', k)
print(f'IN BIB BUT NEVER CITED ({len(unused)}):')
for k in unused: print(' ', k)
