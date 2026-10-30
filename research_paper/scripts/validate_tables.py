import re
from pathlib import Path

def extract_braced(text, start_pos):
    if text[start_pos] != '{':
        return None, start_pos
    depth = 0
    content = []
    for i in range(start_pos, len(text)):
        if text[i] == '{':
            depth += 1
            if depth > 1:
                content.append('{')
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return "".join(content), i + 1
            else:
                content.append('}')
        else:
            if depth >= 1:
                content.append(text[i])
    return None, len(text)

def parse_col_count(spec):
    cleaned = re.sub(r'[@><]\{[^}]*\}', '', spec)
    cleaned = re.sub(r'\|', '', cleaned)
    cleaned = re.sub(r'[pmb]\{[^}]*\}', 'P', cleaned)
    cols = re.findall(r'[lcrXP]', cleaned)
    return len(cols)

def check_file(f):
    content = f.read_text(encoding='utf-8')
    pos = 0
    discrepancies = 0
    while True:
        m = re.search(r'\\begin\{(tabular\*?|tabularx)\}', content[pos:])
        if not m:
            break
        env_type = m.group(1)
        cur = pos + m.end()
        if env_type in ['tabular*', 'tabularx']:
            while cur < len(content) and content[cur].isspace():
                cur += 1
            if cur < len(content) and content[cur] == '{':
                _, cur = extract_braced(content, cur)
        
        while cur < len(content) and content[cur].isspace():
            cur += 1
        if cur < len(content) and content[cur] == '{':
            spec, cur = extract_braced(content, cur)
        else:
            pos = cur
            continue
        
        end_str = f'\\end{{{env_type}}}'
        end_idx = content.find(end_str, cur)
        body = content[cur:end_idx] if end_idx != -1 else content[cur:]
        pos = end_idx + len(end_str) if end_idx != -1 else len(content)
            
        expected_cols = parse_col_count(spec)
        # Regex split on row terminators: \\ but not \% or \hline
        rows = re.split(r'(?<!\\)\\\\\s*(?:\r?\n|$)', body)
        for r_idx, row in enumerate(rows[:-1], 1):
            clean = re.sub(r'(?<!\\)%.*$', '', row, flags=re.MULTILINE).strip()
            clean = re.sub(r'\\(toprule|midrule|bottomrule|hline)', '', clean).strip()
            if not clean:
                continue
            
            multi_spans = [int(s) for s in re.findall(r'\\multicolumn\{(\d+)\}', clean)]
            clean_no_multi = re.sub(r'\\multicolumn\{\d+\}\{[^}]*\}\{(?:[^{}]*|\{[^{}]*\})*\}', 'COL', clean)
            
            amps = len(re.findall(r'(?<!\\)&', clean_no_multi))
            actual_cols = amps + 1 + sum(s - 1 for s in multi_spans)
            if actual_cols != expected_cols:
                print(f'{f.name} (spec "{spec}", exp {expected_cols}): Row {r_idx} has {actual_cols} cols')
                print(f'   Row: {clean[:100]}')
                discrepancies += 1
    return discrepancies

def main():
    files = list(Path('research_paper/tables').glob('*.tex')) + [Path('research_paper/main.tex')]
    total = 0
    for f in files:
        total += check_file(f)
    print(f"Tabular verification finished. Found {total} discrepancies.")

if __name__ == '__main__':
    main()
