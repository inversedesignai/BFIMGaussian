"""Scan cited bib entries for red flags suggesting fabrication or incomplete data."""
import re

with open(r'f:\My Drive\BFIMGaussian\doc\paper\paper.tex') as f:
    tex = f.read()
cited = set()
for m in re.finditer(r'\\cite\{([^}]+)\}', tex):
    for key in m.group(1).split(','):
        cited.add(key.strip())

# Parse bib entries into dicts
with open(r'f:\My Drive\BFIMGaussian\doc\paper\paper.bib') as f:
    bib = f.read()

entries = {}
for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}', bib, re.DOTALL):
    etype = m.group(1).lower()
    key = m.group(2).strip()
    body = m.group(3)
    fields = {'__etype__': etype}
    for fm in re.finditer(r'(\w+)\s*=\s*\{(.*?)\}\s*[,\n]', body, re.DOTALL):
        fields[fm.group(1).lower()] = fm.group(2).strip()
    entries[key] = fields

# Red flags
def check_entry(key, fields):
    flags = []
    author = fields.get('author', '')
    title = fields.get('title', '')
    journal = fields.get('journal', '').strip()
    if 'others' in author and author.count('and') <= 2:
        flags.append(f'"and others" with few authors: {author!r}')
    if '[CHECK]' in title or '[CHECK]' in author or '[CHECK]' in journal:
        flags.append('[CHECK] marker')
    if not journal and fields['__etype__'] == 'article':
        flags.append('empty journal (@article)')
    if journal in ('preprint', 'arXiv', '[CHECK]'):
        flags.append(f'suspicious journal: {journal!r}')
    if not title:
        flags.append('missing title')
    if not author:
        flags.append('missing author')
    if 'doi' not in fields and 'arxiv' not in str(fields).lower() and fields['__etype__'] == 'article':
        pass  # not really a flag; many legit entries don't have DOI
    return flags

print('CITED ENTRIES WITH RED FLAGS:')
print('-' * 70)
count = 0
for key in sorted(cited):
    if key not in entries:
        continue
    flags = check_entry(key, entries[key])
    if flags:
        count += 1
        print(f'\n{key}:')
        for f in flags:
            print(f'  ! {f}')
        if 'author' in entries[key]:
            print(f'  author: {entries[key]["author"]}')
        if 'title' in entries[key]:
            print(f'  title:  {entries[key]["title"]}')
        if 'journal' in entries[key]:
            print(f'  journal:{entries[key]["journal"]}')
        print(f'  year:   {entries[key].get("year", "?")}')
print(f'\n=== Total flagged: {count} of {len(cited)} cited entries ===')
