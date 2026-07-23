<#
.SYNOPSIS
  Detect unintended styling in commit messages, per the /commit skill rules.

.DESCRIPTION
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

  Exit code: 1 if any error, otherwise 0 (-Strict: 1 if any warning too)

.EXAMPLE
  .\check-commit-style.ps1                     # check the HEAD commit message
  .\check-commit-style.ps1 'HEAD^'             # check a specific commit
  .\check-commit-style.ps1 -Path msg.txt       # check a draft file (UTF-8) before committing
  .\check-commit-style.ps1 -Message "Fix bug"  # check a string directly
  .\check-commit-style.ps1 -Path .git\COMMIT_EDITMSG   # usable as a commit-msg hook
#>
[CmdletBinding(DefaultParameterSetName = 'Commit')]
param(
    [Parameter(ParameterSetName = 'Path', Mandatory = $true)]
    [string]$Path,

    [Parameter(ParameterSetName = 'Message', Mandatory = $true)]
    [string]$Message,

    [Parameter(ParameterSetName = 'Commit', Position = 0)]
    [string]$Commit = 'HEAD',

    [switch]$Strict,
    [switch]$AsJson
)

$ErrorActionPreference = 'Stop'

# ---- Obtain the message text -------------------------------------------------

$fromEditMsgFile = $false
$text = ''
switch ($PSCmdlet.ParameterSetName) {
    'Path' {
        $resolved = (Resolve-Path -LiteralPath $Path).ProviderPath
        $text = [System.IO.File]::ReadAllText($resolved, [System.Text.Encoding]::UTF8)
        $fromEditMsgFile = $true   # skip git comment lines (^#) and everything after scissors
    }
    'Message' {
        $text = $Message
    }
    'Commit' {
        $prevEnc = [Console]::OutputEncoding
        try {
            try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
            $raw = git log -1 --format=%B $Commit
            if ($LASTEXITCODE -ne 0) { Write-Error "git log failed (ref: $Commit)"; exit 2 }
            $text = @($raw) -join "`n"
        }
        finally {
            try { [Console]::OutputEncoding = $prevEnc } catch {}
        }
    }
}

# ---- Split into lines (keep original line numbers, drop git comments) ---------

$allLines = (($text -replace "`r`n", "`n") -replace "`r", "`n") -split "`n"
$entries = New-Object System.Collections.Generic.List[object]
for ($i = 0; $i -lt $allLines.Count; $i++) {
    $t = $allLines[$i]
    if ($fromEditMsgFile) {
        if ($t -match '^#.*>8') { break }    # verbose diff after the scissors line
        if ($t -match '^#') { continue }     # git comment line
    }
    $entries.Add([pscustomobject]@{ N = $i + 1; Text = $t })
}
while ($entries.Count -gt 0 -and $entries[$entries.Count - 1].Text -match '^\s*$') {
    $entries.RemoveAt($entries.Count - 1)
}

# ---- Helpers -------------------------------------------------------------------

$findings = New-Object System.Collections.Generic.List[object]
function Add-Finding {
    param(
        [int]$Line,
        [ValidateSet('error', 'warning', 'info')][string]$Severity,
        [string]$Rule,
        [string]$Snippet,
        [string]$Advice
    )
    $findings.Add([pscustomobject]@{
        Line = $Line; Severity = $Severity; Rule = $Rule; Snippet = $Snippet; Advice = $Advice
    })
}

function Get-Context {
    param([string]$Line, [System.Text.RegularExpressions.Match]$M)
    $start = [Math]::Max(0, $M.Index - 10)
    $end = [Math]::Min($Line.Length, $M.Index + $M.Length + 10)
    $ctx = $Line.Substring($start, $end - $start)
    if ($start -gt 0) { $ctx = '...' + $ctx }
    if ($end -lt $Line.Length) { $ctx += '...' }
    return $ctx
}

function Get-Trunc {
    param([string]$s)
    if ($s.Length -gt 60) { $s.Substring(0, 60) + '...' } else { $s }
}

# Index of the first line of the trailing trailer paragraph (the entry count if none).
# The trailer paragraph is the last block of the message when every one of its lines is
# a "Token: value" trailer, so the body checks can skip it.
function Get-TrailerBlockStart {
    param($Entries)
    $i = $Entries.Count - 1
    while ($i -ge 0 -and $Entries[$i].Text -notmatch '^\s*$') { $i-- }
    $start = $i + 1
    # The whole message being one block means it is the subject, not a trailer paragraph
    if ($start -eq 0 -or $start -ge $Entries.Count) { return $Entries.Count }
    for ($j = $start; $j -lt $Entries.Count; $j++) {
        if ($Entries[$j].Text -notmatch '^[A-Za-z][A-Za-z0-9-]*:\s') { return $Entries.Count }
    }
    return $start
}

# Thresholds beyond which a body reads as an unsummarized dump rather than a summary
$maxBulletItems = 6
$maxBulletChars = 180

# Characters that must be replaced with an ASCII equivalent.
# Built from [char] codes so the script itself stays pure ASCII.
$replaceMap = @{}
foreach ($pair in @(
    @(0x2192, '->'),  @(0x21D2, '=>'),  @(0x2190, '<-'),  @(0x21D0, '<='),   # arrows
    @(0x201C, '"'),   @(0x201D, '"'),   @(0x2018, "'"),   @(0x2019, "'"),    # smart quotes
    @(0x2026, '...'), @(0x2014, '-'),   @(0x2013, '-'),   @(0x2015, '-'),    # ellipsis / dashes
    @(0x00D7, 'x'),   @(0x2212, '-'),   @(0x2022, '-'),   @(0x00B1, '+/-'),  # multiply / minus / bullet
    @(0x2260, '!='),  @(0x2264, '<='),  @(0x2265, '>='),                     # comparison signs
    @(0x3001, ', '),  @(0x3002, '. '),  @(0x300C, '"'),   @(0x300D, '"'),    # CJK comma / period / corner brackets
    @(0xFF08, '('),   @(0xFF09, ')'),   @(0xFF1A, ':'),   @(0xFF1B, ';'),    # fullwidth parens / colon / semicolon
    @(0xFF01, '!'),   @(0xFF1F, '?'),   @(0xFF0C, ', '),  @(0xFF0E, '. ')    # fullwidth ! ? , .
)) { $replaceMap[[char]$pair[0]] = $pair[1] }

# Invisible characters (always an error)
$invisibleMap = @{}
foreach ($pair in @(
    @(0x00A0, 'NBSP'), @(0x3000, 'IDEOGRAPHIC SPACE'), @(0x200B, 'ZERO WIDTH SPACE'),
    @(0x200C, 'ZWNJ'), @(0x200D, 'ZWJ'), @(0xFEFF, 'BOM/ZWNBSP')
)) { $invisibleMap[[char]$pair[0]] = $pair[1] }

# Regexes are also built at runtime (hiragana / katakana / CJK ideographs / compat ideographs)
$jpScript = ('[{0}-{1}{2}-{3}{4}-{5}]' -f `
    [char]0x3040, [char]0x30FF, [char]0x3400, [char]0x9FFF, [char]0xF900, [char]0xFAFF)
$nonAsciiPattern = ('[^{0}-{1}]' -f [char]0x0009, [char]0x007E)   # outside TAB-~ = non-ASCII

# ---- Checks --------------------------------------------------------------------

if ($entries.Count -eq 0) {
    Add-Finding 0 'error' 'empty' '' 'The message is empty'
}
else {
    # --- Subject line ---
    $subjectEntry = $entries[0]
    $s = $subjectEntry.Text
    if ($s.Length -gt 70) {
        Add-Finding $subjectEntry.N 'error' 'subject-length' (Get-Trunc $s) "Subject is $($s.Length) chars; keep it within 70"
    }
    if ($s -match $jpScript) {
        Add-Finding $subjectEntry.N 'warning' 'subject-english' (Get-Trunc $s) 'Write the subject in English imperative form'
    }
    if ($s -notmatch '^[A-Z][A-Za-z]') {
        Add-Finding $subjectEntry.N 'warning' 'subject-imperative' (Get-Trunc $s) 'Start the subject with an English imperative verb (Add / Update / Fix ...)'
    }
    else {
        $firstWord = ($s -split '\s+')[0]
        if ($firstWord -match '^(Added|Adding|Adds|Updated|Updating|Updates|Fixed|Fixing|Fixes|Removed|Removing|Removes|Changed|Changing|Changes|Implemented|Implementing|Implements|Refactored|Refactoring|Refactors|Created|Creating|Creates|Improved|Improving|Improves|Renamed|Renaming|Renames|Moved|Moving|Moves|Deleted|Deleting|Deletes|Introduced|Introducing|Introduces|Enhanced|Enhancing|Enhances)$') {
            Add-Finding $subjectEntry.N 'warning' 'subject-imperative' $firstWord "'$firstWord' may not be imperative; use the base form (Add / Update / Fix ...)"
        }
    }

    # --- Blank line right after the subject ---
    if ($entries.Count -ge 2 -and $entries[1].Text -notmatch '^\s*$') {
        Add-Finding $entries[1].N 'error' 'blank-line' (Get-Trunc $entries[1].Text) 'Insert a blank line between the subject and the body'
    }

    # --- Per-line checks ---
    $hasAnyJapanese = $false
    $prevIsBullet = $false

    for ($k = 0; $k -lt $entries.Count; $k++) {
        $e = $entries[$k]
        $line = $e.Text
        $isBlank = $line -match '^\s*$'

        # Folded / nested bullet detection (body only)
        if ($k -gt 0) {
            if ($isBlank) {
                $prevIsBullet = $false
            }
            elseif ($line -match '^[-*]\s') {
                $prevIsBullet = $true
            }
            elseif ($prevIsBullet -and $line -match '^\s+\S') {
                Add-Finding $e.N 'error' 'bullet-fold' (Get-Trunc $line) 'Do not wrap a bullet item onto multiple lines (or nest); keep it on one line or split the item'
                # keep $prevIsBullet (a fold can span multiple lines)
            }
            else {
                $prevIsBullet = $false
            }
        }

        if ($isBlank) { continue }

        # Tilde (GFM strikethrough)
        foreach ($m in [regex]::Matches($line, '~+')) {
            Add-Finding $e.N 'error' 'tilde' (Get-Context $line $m) 'Tilde renders as strikethrough in GFM; use about/roughly/approx. for "approximately" and hyphen ranges like 5-10'
        }

        # #<digits> (Issue/PR auto-link)
        foreach ($m in [regex]::Matches($line, '#\d+')) {
            Add-Finding $e.N 'error' 'issue-ref' (Get-Context $line $m) 'GitHub auto-links this to an Issue/PR; drop the numeric reference and describe the change in words'
        }

        # Leading # (GFM heading) - not applicable to draft files (git strips comments)
        if (-not $fromEditMsgFile -and $line -match '^#{1,6}\s') {
            Add-Finding $e.N 'warning' 'md-heading' (Get-Trunc $line) 'A leading # renders as a GFM heading; do not start a line with it'
        }

        # References to local-only artifacts
        if ($line -match '(?i)(?<![\w.-])tmp[/\\]') {
            Add-Finding $e.N 'warning' 'local-ref' (Get-Trunc $line) 'Possible reference to a local-only artifact; if third parties cannot reach it, describe the content inline instead'
        }

        # Non-ASCII characters
        $hasJp = $line -match $jpScript
        if ($hasJp) { $hasAnyJapanese = $true }
        foreach ($m in [regex]::Matches($line, $nonAsciiPattern)) {
            $ch = $m.Value[0]
            $code = [int]$ch
            if ($code -lt 0x20) { continue }   # control chars (CR etc.) are out of scope
            if ($invisibleMap.ContainsKey($ch)) {
                Add-Finding $e.N 'error' 'invisible' ("U+{0:X4} ({1})" -f $code, $invisibleMap[$ch]) 'Invisible character; replace with a regular space or remove'
            }
            elseif ($replaceMap.ContainsKey($ch)) {
                if ($hasJp -and $code -ge 0x3000) {
                    continue   # CJK punctuation inside Japanese text is tolerated
                }
                $sev = if ($hasJp) { 'warning' } else { 'error' }
                Add-Finding $e.N $sev 'non-ascii' (Get-Context $line $m) ("Replace U+{0:X4} `"{1}`" with ASCII `"{2}`"" -f $code, $m.Value, $replaceMap[$ch])
            }
            elseif ($ch -match $jpScript) {
                continue   # Japanese body text is tolerated (aggregated as info below)
            }
            elseif ($hasJp -and $code -ge 0x3000) {
                continue   # CJK symbols / fullwidth forms inside Japanese text are tolerated
            }
            else {
                Add-Finding $e.N 'warning' 'non-ascii' ("U+{0:X4} `"{1}`"" -f $code, $m.Value) 'Consider an ASCII substitute'
            }
        }
    }

    if ($hasAnyJapanese) {
        Add-Finding 0 'info' 'japanese' '' 'Contains Japanese text; allowed when the explanation requires it (prefer-ASCII rule)'
    }

    # --- Body structure: bullets only, and summarized enough to read as one ---
    # The body starts after the subject, skipping the line the blank-line rule already reported.
    $bodyStart = if ($entries.Count -ge 2 -and $entries[1].Text -notmatch '^\s*$') { 2 } else { 1 }
    $bodyEnd = Get-TrailerBlockStart $entries
    $bullets = New-Object System.Collections.Generic.List[object]
    $inProse = $false
    $afterBullet = $false

    for ($k = $bodyStart; $k -lt $bodyEnd; $k++) {
        $e = $entries[$k]
        $line = $e.Text
        if ($line -match '^\s*$') {
            $inProse = $false
            $afterBullet = $false
        }
        elseif ($line -match '^[-*]\s') {
            $bullets.Add($e)
            $inProse = $false
            $afterBullet = $true
        }
        elseif ($afterBullet -and $line -match '^\s+\S') {
            continue   # a folded bullet, already reported as bullet-fold
        }
        elseif (-not $inProse) {
            Add-Finding $e.N 'error' 'body-bullets' (Get-Trunc $line) 'Write the body as "- " bullet items; a prose paragraph is not the expected format'
            $inProse = $true   # report a paragraph once, not once per line
        }
    }

    if ($bullets.Count -gt $maxBulletItems) {
        Add-Finding $bullets[$maxBulletItems].N 'info' 'bullet-count' '' "$($bullets.Count) bullet items; summarize further, or split the change into separate commits ($maxBulletItems or fewer reads as a summary)"
    }
    foreach ($b in $bullets) {
        if ($b.Text.Length -gt $maxBulletChars) {
            Add-Finding $b.N 'info' 'bullet-length' (Get-Trunc $b.Text) "Bullet is $($b.Text.Length) chars; keep it to what / why / how in one sentence, or split it into separate items"
        }
    }

    # --- Co-Authored-By trailer ---
    # Accepted forms: a bare "Claude", or "Claude" followed by a model name such as
    # "Claude Opus 4.8 (1M context)", both at the same address.
    $expectedTrailer = 'Co-Authored-By: Claude <noreply@anthropic.com>'
    $acceptedTrailer = '(?im)^co-authored-by:[ \t]*Claude(?:[ \t]+\S[^<>]*?)?[ \t]*<noreply@anthropic\.com>[ \t]*$'
    $bodyText = @($entries | ForEach-Object { $_.Text }) -join "`n"
    if ($bodyText -notmatch $acceptedTrailer) {
        if ($bodyText -match '(?im)^co-authored-by:') {
            Add-Finding 0 'info' 'trailer' '' "Co-Authored-By trailer differs from the accepted forms ($expectedTrailer, or the same address with a model name such as `"Claude Opus 4.8 (1M context)`")"
        }
        else {
            Add-Finding 0 'warning' 'trailer' '' "Missing trailer: $expectedTrailer"
        }
    }
}

# ---- Output --------------------------------------------------------------------

$errCount = @($findings | Where-Object { $_.Severity -eq 'error' }).Count
$warnCount = @($findings | Where-Object { $_.Severity -eq 'warning' }).Count

if ($AsJson) {
    if ($findings.Count -eq 0) { Write-Output '[]' }
    else { Write-Output (ConvertTo-Json -InputObject $findings.ToArray() -Depth 3) }
}
else {
    if ($findings.Count -eq 0) {
        Write-Output 'OK: no style violations found'
    }
    else {
        $order = @{ error = 0; warning = 1; info = 2 }
        foreach ($f in ($findings | Sort-Object { $order[$_.Severity] }, { $_.Line })) {
            $loc = if ($f.Line -gt 0) { "L$($f.Line)" } else { '-' }
            $snip = if ($f.Snippet) { " `"$($f.Snippet)`"" } else { '' }
            Write-Output ("{0,-4} [{1,-7}] {2,-18}{3} : {4}" -f $loc, $f.Severity, $f.Rule, $snip, $f.Advice)
        }
        Write-Output ''
        Write-Output ("Result: {0} error(s) / {1} warning(s)" -f $errCount, $warnCount)
    }
}

if ($errCount -gt 0) { exit 1 }
if ($Strict -and $warnCount -gt 0) { exit 1 }
exit 0
