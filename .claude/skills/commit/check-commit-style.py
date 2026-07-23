#!/usr/bin/env python3
"""Detect unintended styling in commit messages, per the /commit skill rules.

Cross-platform (Python 3.6+, stdlib only) port of check-commit-style.ps1.

Rules checked (derived from skills/commit/SKILL.md):
  subject-length     : subject line exceeds 70 chars (error)
  subject-imperative : subject does not look like English imperative (warning)
  subject-english    : subject contains Japanese text (warning)
  blank-line         : missing blank line between subject and body (error)
  non-ascii          : symbols that should be replaced with ASCII equivalents
                       (arrows, smart quotes, ellipsis, dashes, etc.);
                       Japanese body text itself is tolerated (aggregated as info)
  invisible          : invisible chars such as NBSP / ideographic space / zero-width (error)
  body-bullets       : body written as a prose paragraph instead of "- " bullets (error)
  bullet-fold        : bullet item wrapped onto multiple lines, or nested (error)
  tilde              : tilde renders as strikethrough in GitHub-flavored Markdown (error)
  issue-ref          : #<digits> auto-links to unrelated GitHub Issues/PRs (error)
  md-heading         : leading # renders as a GFM heading (warning)
  local-ref          : suspected reference to local-only artifacts like tmp/ (warning)
  trailer            : Co-Authored-By trailer presence (warning)
  bullet-count       : so many bullet items that the message is not a summary (info)
  bullet-length      : a single bullet item too long to read as a summary (info)

Exit code: 1 if any error, otherwise 0 (--strict: 1 if any warning too; 2 on git failure)

Examples:
  check-commit-style.py                       # check the HEAD commit message
  check-commit-style.py 'HEAD^'               # check a specific commit
  check-commit-style.py --path msg.txt        # check a draft file (UTF-8) before committing
  check-commit-style.py --message "Fix bug"   # check a string directly
  check-commit-style.py --path .git/COMMIT_EDITMSG   # usable as a commit-msg hook
"""

import argparse
import json
import re
import subprocess
import sys

# Characters that must be replaced with an ASCII equivalent.
# Built from code points so this source file stays pure ASCII.
REPLACE_MAP = {chr(c): r for c, r in [
    (0x2192, '->'),  (0x21D2, '=>'),  (0x2190, '<-'),  (0x21D0, '<='),   # arrows
    (0x201C, '"'),   (0x201D, '"'),   (0x2018, "'"),   (0x2019, "'"),    # smart quotes
    (0x2026, '...'), (0x2014, '-'),   (0x2013, '-'),   (0x2015, '-'),    # ellipsis / dashes
    (0x00D7, 'x'),   (0x2212, '-'),   (0x2022, '-'),   (0x00B1, '+/-'),  # multiply / minus / bullet
    (0x2260, '!='),  (0x2264, '<='),  (0x2265, '>='),                    # comparison signs
    (0x3001, ', '),  (0x3002, '. '),  (0x300C, '"'),   (0x300D, '"'),    # CJK comma / period / corner brackets
    (0xFF08, '('),   (0xFF09, ')'),   (0xFF1A, ':'),   (0xFF1B, ';'),    # fullwidth parens / colon / semicolon
    (0xFF01, '!'),   (0xFF1F, '?'),   (0xFF0C, ', '),  (0xFF0E, '. '),   # fullwidth ! ? , .
]}

# Invisible characters (always an error)
INVISIBLE_MAP = {chr(c): n for c, n in [
    (0x00A0, 'NBSP'), (0x3000, 'IDEOGRAPHIC SPACE'), (0x200B, 'ZERO WIDTH SPACE'),
    (0x200C, 'ZWNJ'), (0x200D, 'ZWJ'), (0xFEFF, 'BOM/ZWNBSP'),
]}

# Hiragana / katakana / CJK ideographs / compat ideographs
JP_SCRIPT = re.compile('[{}-{}{}-{}{}-{}]'.format(
    chr(0x3040), chr(0x30FF), chr(0x3400), chr(0x9FFF), chr(0xF900), chr(0xFAFF)))
# Outside TAB-~ = non-ASCII
NON_ASCII = re.compile('[^{}-{}]'.format(chr(0x09), chr(0x7E)))

NON_IMPERATIVE = re.compile(
    r'^(Added|Adding|Adds|Updated|Updating|Updates|Fixed|Fixing|Fixes'
    r'|Removed|Removing|Removes|Changed|Changing|Changes'
    r'|Implemented|Implementing|Implements|Refactored|Refactoring|Refactors'
    r'|Created|Creating|Creates|Improved|Improving|Improves'
    r'|Renamed|Renaming|Renames|Moved|Moving|Moves|Deleted|Deleting|Deletes'
    r'|Introduced|Introducing|Introduces|Enhanced|Enhancing|Enhances)$')

EXPECTED_TRAILER = 'Co-Authored-By: Claude <noreply@anthropic.com>'

# Accepted Co-Authored-By forms: a bare "Claude", or "Claude" followed by a model
# name such as "Claude Opus 4.8 (1M context)", both at the same address.
ACCEPTED_TRAILER = re.compile(
    r'^co-authored-by:[ \t]*Claude(?:[ \t]+\S[^<>]*?)?[ \t]*<noreply@anthropic\.com>[ \t]*$',
    re.IGNORECASE | re.MULTILINE)

BULLET = re.compile(r'^[-*]\s')
# A git trailer line such as "Co-Authored-By: ..." or "Signed-off-by: ..."
TRAILER_LINE = re.compile(r'^[A-Za-z][A-Za-z0-9-]*:\s')

# Thresholds beyond which a body reads as an unsummarized dump rather than a summary
MAX_BULLET_ITEMS = 6
MAX_BULLET_CHARS = 180

SEVERITY_ORDER = {'error': 0, 'warning': 1, 'info': 2}


def get_context(line, m):
    start = max(0, m.start() - 10)
    end = min(len(line), m.end() + 10)
    ctx = line[start:end]
    if start > 0:
        ctx = '...' + ctx
    if end < len(line):
        ctx += '...'
    return ctx


def trunc(s, limit=60):
    return s[:limit] + '...' if len(s) > limit else s


def get_message_text(args):
    """Return (text, from_edit_msg_file). Exits with code 2 on git failure."""
    if args.path is not None:
        with open(args.path, encoding='utf-8-sig', errors='replace') as f:
            return f.read(), True
    if args.message is not None:
        return args.message, False
    proc = subprocess.run(
        ['git', 'log', '-1', '--format=%B', args.commit],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        sys.stderr.write('git log failed (ref: {}): {}\n'.format(
            args.commit, proc.stderr.decode('utf-8', errors='replace').strip()))
        sys.exit(2)
    return proc.stdout.decode('utf-8', errors='replace'), False


def split_entries(text, from_edit_msg_file):
    """Split into (line_number, text) pairs, dropping git comment lines."""
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    entries = []
    for i, t in enumerate(lines):
        if from_edit_msg_file:
            if re.match(r'^#.*>8', t):   # verbose diff after the scissors line
                break
            if t.startswith('#'):        # git comment line
                continue
        entries.append((i + 1, t))
    while entries and not entries[-1][1].strip():
        entries.pop()
    return entries


def trailer_block_start(entries):
    """Index of the first line of the trailing trailer paragraph (len(entries) if none).

    The trailer paragraph is the last block of the message when every one of its
    lines is a "Token: value" trailer, so the body checks can skip it.
    """
    i = len(entries) - 1
    while i >= 0 and entries[i][1].strip():
        i -= 1
    start = i + 1
    if start == 0 or start >= len(entries):
        return len(entries)   # the whole message is one block: that is the subject, not trailers
    if all(TRAILER_LINE.match(t) for _, t in entries[start:]):
        return start
    return len(entries)


def check(entries, from_edit_msg_file):
    findings = []

    def add(line, severity, rule, snippet, advice):
        findings.append({'Line': line, 'Severity': severity, 'Rule': rule,
                         'Snippet': snippet, 'Advice': advice})

    if not entries:
        add(0, 'error', 'empty', '', 'The message is empty')
        return findings

    # --- Subject line ---
    subject_n, s = entries[0]
    if len(s) > 70:
        add(subject_n, 'error', 'subject-length', trunc(s),
            'Subject is {} chars; keep it within 70'.format(len(s)))
    if JP_SCRIPT.search(s):
        add(subject_n, 'warning', 'subject-english', trunc(s),
            'Write the subject in English imperative form')
    if not re.match(r'^[A-Z][A-Za-z]', s):
        add(subject_n, 'warning', 'subject-imperative', trunc(s),
            'Start the subject with an English imperative verb (Add / Update / Fix ...)')
    else:
        first_word = s.split()[0]
        if NON_IMPERATIVE.match(first_word):
            add(subject_n, 'warning', 'subject-imperative', first_word,
                "'{}' may not be imperative; use the base form (Add / Update / Fix ...)".format(first_word))

    # --- Blank line right after the subject ---
    if len(entries) >= 2 and entries[1][1].strip():
        add(entries[1][0], 'error', 'blank-line', trunc(entries[1][1]),
            'Insert a blank line between the subject and the body')

    # --- Per-line checks ---
    has_any_japanese = False
    prev_is_bullet = False

    for k, (n, line) in enumerate(entries):
        is_blank = not line.strip()

        # Folded / nested bullet detection (body only)
        if k > 0:
            if is_blank:
                prev_is_bullet = False
            elif re.match(r'^[-*]\s', line):
                prev_is_bullet = True
            elif prev_is_bullet and re.match(r'^\s+\S', line):
                add(n, 'error', 'bullet-fold', trunc(line),
                    'Do not wrap a bullet item onto multiple lines (or nest); '
                    'keep it on one line or split the item')
                # keep prev_is_bullet (a fold can span multiple lines)
            else:
                prev_is_bullet = False

        if is_blank:
            continue

        # Tilde (GFM strikethrough)
        for m in re.finditer(r'~+', line):
            add(n, 'error', 'tilde', get_context(line, m),
                'Tilde renders as strikethrough in GFM; use about/roughly/approx. '
                'for "approximately" and hyphen ranges like 5-10')

        # #<digits> (Issue/PR auto-link)
        for m in re.finditer(r'#\d+', line):
            add(n, 'error', 'issue-ref', get_context(line, m),
                'GitHub auto-links this to an Issue/PR; drop the numeric reference '
                'and describe the change in words')

        # Leading # (GFM heading) - not applicable to draft files (git strips comments)
        if not from_edit_msg_file and re.match(r'^#{1,6}\s', line):
            add(n, 'warning', 'md-heading', trunc(line),
                'A leading # renders as a GFM heading; do not start a line with it')

        # References to local-only artifacts
        if re.search(r'(?<![\w.-])tmp[/\\]', line, re.IGNORECASE):
            add(n, 'warning', 'local-ref', trunc(line),
                'Possible reference to a local-only artifact; if third parties '
                'cannot reach it, describe the content inline instead')

        # Non-ASCII characters
        has_jp = bool(JP_SCRIPT.search(line))
        if has_jp:
            has_any_japanese = True
        for m in NON_ASCII.finditer(line):
            ch = m.group()
            code = ord(ch)
            if code < 0x20:   # control chars (CR etc.) are out of scope
                continue
            if ch in INVISIBLE_MAP:
                add(n, 'error', 'invisible',
                    'U+{:04X} ({})'.format(code, INVISIBLE_MAP[ch]),
                    'Invisible character; replace with a regular space or remove')
            elif ch in REPLACE_MAP:
                if has_jp and code >= 0x3000:
                    continue   # CJK punctuation inside Japanese text is tolerated
                sev = 'warning' if has_jp else 'error'
                add(n, sev, 'non-ascii', get_context(line, m),
                    'Replace U+{:04X} "{}" with ASCII "{}"'.format(code, ch, REPLACE_MAP[ch]))
            elif JP_SCRIPT.match(ch):
                continue   # Japanese body text is tolerated (aggregated as info below)
            elif has_jp and code >= 0x3000:
                continue   # CJK symbols / fullwidth forms inside Japanese text are tolerated
            else:
                add(n, 'warning', 'non-ascii',
                    'U+{:04X} "{}"'.format(code, ch),
                    'Consider an ASCII substitute')

    if has_any_japanese:
        add(0, 'info', 'japanese', '',
            'Contains Japanese text; allowed when the explanation requires it (prefer-ASCII rule)')

    # --- Body structure: bullets only, and summarized enough to read as one ---
    # The body starts after the subject, skipping the line the blank-line rule already reported.
    body_start = 2 if (len(entries) >= 2 and entries[1][1].strip()) else 1
    body_end = trailer_block_start(entries)
    bullets = []
    in_prose = False
    after_bullet = False
    for n, line in entries[body_start:body_end]:
        if not line.strip():
            in_prose = False
            after_bullet = False
        elif BULLET.match(line):
            bullets.append((n, line))
            in_prose = False
            after_bullet = True
        elif after_bullet and re.match(r'^\s+\S', line):
            continue   # a folded bullet, already reported as bullet-fold
        elif not in_prose:
            add(n, 'error', 'body-bullets', trunc(line),
                'Write the body as "- " bullet items; a prose paragraph is not the expected format')
            in_prose = True   # report a paragraph once, not once per line

    if len(bullets) > MAX_BULLET_ITEMS:
        add(bullets[MAX_BULLET_ITEMS][0], 'info', 'bullet-count', '',
            '{} bullet items; summarize further, or split the change into separate commits '
            '({} or fewer reads as a summary)'.format(len(bullets), MAX_BULLET_ITEMS))
    for n, line in bullets:
        if len(line) > MAX_BULLET_CHARS:
            add(n, 'info', 'bullet-length', trunc(line),
                'Bullet is {} chars; keep it to what / why / how in one sentence, '
                'or split it into separate items'.format(len(line)))

    # --- Co-Authored-By trailer ---
    body_text = '\n'.join(t for _, t in entries)
    if not ACCEPTED_TRAILER.search(body_text):
        if re.search(r'^co-authored-by:', body_text, re.IGNORECASE | re.MULTILINE):
            add(0, 'info', 'trailer', '',
                'Co-Authored-By trailer differs from the accepted forms ({}, or the same '
                'address with a model name such as "Claude Opus 4.8 (1M context)")'.format(EXPECTED_TRAILER))
        else:
            add(0, 'warning', 'trailer', '', 'Missing trailer: {}'.format(EXPECTED_TRAILER))

    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Detect unintended styling in commit messages (see /commit skill rules).')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--path', help='check a message draft file (UTF-8); '
                                       'git comment lines and scissors are ignored')
    source.add_argument('--message', help='check a message string directly')
    parser.add_argument('commit', nargs='?', default='HEAD',
                        help='commit ref to check via git log (default: HEAD)')
    parser.add_argument('--strict', action='store_true',
                        help='exit 1 on warnings as well as errors')
    parser.add_argument('--json', action='store_true', dest='as_json',
                        help='emit findings as JSON')
    args = parser.parse_args(argv)

    # Avoid UnicodeEncodeError on consoles with narrow code pages (e.g. cp932)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(errors='replace')
            except Exception:
                pass

    text, from_edit_msg_file = get_message_text(args)
    entries = split_entries(text, from_edit_msg_file)
    findings = check(entries, from_edit_msg_file)

    err_count = sum(1 for f in findings if f['Severity'] == 'error')
    warn_count = sum(1 for f in findings if f['Severity'] == 'warning')

    if args.as_json:
        print(json.dumps(findings, ensure_ascii=True, indent=2))
    elif not findings:
        print('OK: no style violations found')
    else:
        ordered = sorted(findings, key=lambda f: (SEVERITY_ORDER[f['Severity']], f['Line']))
        for f in ordered:
            loc = 'L{}'.format(f['Line']) if f['Line'] > 0 else '-'
            snip = ' "{}"'.format(f['Snippet']) if f['Snippet'] else ''
            print('{:<4} [{:<7}] {:<18}{} : {}'.format(
                loc, f['Severity'], f['Rule'], snip, f['Advice']))
        print()
        print('Result: {} error(s) / {} warning(s)'.format(err_count, warn_count))

    if err_count > 0:
        return 1
    if args.strict and warn_count > 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
