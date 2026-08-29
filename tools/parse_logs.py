#!/usr/bin/env python
# aggregate eval_*.json summaries into a markdown table (experiments.md).
#
# Usage:
#   python tools/parse_logs.py exps/m2tdiff                      # print table
#   python tools/parse_logs.py exps/m2tdiff --out docs/experiments.md
#   python tools/parse_logs.py exps/A exps/B --out docs/experiments.md
#
# Scans the given dirs recursively for eval_*.json written by main.py --eval
# (mAP@0.5, ms/frame, FPS, n_parameters, args snapshot). Rows are keyed by the
# experiment tag (parent dir name); re-running updates existing rows in place.

import argparse
import json
import sys
from pathlib import Path


def load_summary(path):
    """Return the parsed eval json or None."""
    if path.suffix == '.json' and path.name.startswith('eval_'):
        try:
            return json.loads(path.read_text())
        except Exception as e:
            print('[warn] skip {}: {}'.format(path, e), file=sys.stderr)
    return None


def config_brief(args):
    """Short human-readable config string derived from the args snapshot."""
    if not isinstance(args, dict):
        return ''
    flags = [name for name, flag in
             (('use_rdqg', 'RDQG'), ('use_mgte', 'MGTE'), ('use_smtd', 'SMTD'))
             if args.get(name)]
    brief = '+'.join(flags) if flags else 'baseline'
    parts = []
    if args.get('use_rdqg'):
        parts.append('T{} K{}'.format(args.get('diffusion_steps'),
                                      args.get('num_diffusion_trajectories')))
    if args.get('use_mgte'):
        parts.append('L{} knn{}'.format(args.get('graph_layers'), args.get('knn_k')))
    if args.get('use_smtd'):
        parts.append('Y{}'.format(args.get('num_experts')))
    if parts:
        brief += ' (' + ' '.join(parts) + ')'
    frames = args.get('frames')
    if frames is None and args.get('num_ref_frames'):
        frames = args['num_ref_frames'] + 1
    if frames:
        brief += ' M{}'.format(frames)
    return brief


def main():
    ap = argparse.ArgumentParser(description='Aggregate eval_*.json into a markdown table')
    ap.add_argument('roots', nargs='+',
                    help='one or more exp dirs (or a root scanned recursively), '
                         'or a single eval_*.json file')
    ap.add_argument('--out', default=None,
                    help='md file to write; rows with the same exp_tag are '
                         'replaced (default: print the table to stdout)')
    args = ap.parse_args()

    rows = []
    seen = set()
    for root in args.roots:
        p = Path(root)
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = sorted(p.rglob('eval_*.json'))
        else:
            print('[warn] not found: {}'.format(root), file=sys.stderr)
            continue
        for f in files:
            data = load_summary(f)
            if data is None:
                continue
            tag = data.get('exp_tag') or f.parent.name
            if tag in seen:
                continue
            seen.add(tag)
            rows.append((tag,
                         str(data.get('dataset', '?')),
                         float(data.get('mAP_50', float('nan'))),
                         float(data.get('ms_per_frame', float('nan'))),
                         float(data.get('fps', float('nan'))),
                         int(data.get('n_parameters', 0)),
                         config_brief(data.get('args') or {})))

    if not rows:
        print('no eval_*.json found under: ' + ', '.join(args.roots), file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: r[0])
    lines = ['| exp_tag | dataset | mAP@0.5 | ms/frame | FPS | params | config |',
             '|---|---|---|---|---|---|---|']
    for tag, ds, ap50, ms, fps, params, brief in rows:
        lines.append('| {} | {} | {:.4f} | {:.1f} | {:.1f} | {:,} | {} |'.format(
            tag, ds, ap50, ms, fps, params, brief))
    table = '\n'.join(lines)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        header = '# Experiment Log\n\n'
        body = out.read_text() if out.exists() else ''
        # replace everything between the markers to keep the table in sync
        start, end = body.find('<!-- exp-table -->'), body.find('<!-- /exp-table -->')
        new_body = '<!-- exp-table -->\n' + table + '\n<!-- /exp-table -->\n'
        if start >= 0 and end > start:
            body = body[:start] + new_body + body[end + len('<!-- /exp-table -->'):]
        else:
            body = header + new_body + (body if body else '')
        out.write_text(body)
        print('wrote {} rows -> {}'.format(len(rows), out))
    else:
        print(table)


if __name__ == '__main__':
    main()
