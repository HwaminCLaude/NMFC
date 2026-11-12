<#
.SYNOPSIS
    Build nmfc:cpu or nmfc:gpu - the only build path on Windows.

.DESCRIPTION
    Same steps as docker/build.sh:
      1. Export a clean tree from the git index to an ASCII path
         (git checkout-index -a -f --prefix=<ctx>/; if the index is empty: HEAD, i.e. the content of
         'git archive HEAD', via 'git read-tree HEAD' into a temporary index file).
         Revision: 'git rev-parse --short=12 HEAD', plus '+index' when the exported index differs from HEAD
         ('git diff --cached --quiet' fails; not added when an empty index made it export HEAD);
         'no-commit' when HEAD does not exist.
      2. Lock:  <python> docker/nmfc_cli.py manifest --write-lock <ctx>   (the exported copy, cwd=<ctx>)
      3. Hash:  NMFC_MANIFEST_SHA256 = sha256(<ctx>/docker/context.lock)
      4. Build: docker build --provenance=false --target <cpu|gpu>
                  --build-arg NMFC_MANIFEST_SHA256=<sha> --build-arg NMFC_MANIFEST_SOURCE=host-export
                  --build-arg NMFC_REVISION=<revision>
                  -t nmfc:<target> <ctx>
      5. Record: docker image inspect -> out/build_<target>.json (Id, RootFS.Layers, Size, Labels, ...)

    Windows PowerShell 5.1 compatible. Binary data is never piped or redirected
    (PowerShell 5.1 re-encodes native byte streams); only text output is captured.
    This file is ASCII-only on purpose: PowerShell 5.1 reads BOM-less scripts in the ANSI code page.

.PARAMETER Target
    cpu or gpu.
.PARAMETER Python
    Interpreter used for step 2. It needs pyyaml and numpy (nmfc_* modules import them at top level).
    Default: 'python'.
.PARAMETER Docker
    docker executable. Default: 'docker'.
.PARAMETER Context
    ASCII export directory. Default: $env:USERPROFILE\nmfc-ctx (e.g. C:\Users\<name>\nmfc-ctx).
    The directory is deleted and re-created; its name must start with 'nmfc-ctx'.
.PARAMETER SkipBuild
    Stop after steps 1-3 (no docker daemon needed). For dry runs.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File docker\build.ps1 -Target cpu
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File docker\build.ps1 -Target gpu -Python C:\path\to\venv\Scripts\python.exe
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('cpu', 'gpu')]
    [string]$Target,
    [string]$Python = 'python',
    [string]$Docker = 'docker',
    [string]$Context = '',
    [switch]$SkipBuild
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

function Write-Step([string]$Message) {
    Write-Host ("[build.ps1] " + $Message)
}

function Invoke-Native {
    # Runs a native command. Text output either streams to the host or is returned (-Capture).
    # stderr is never redirected (PowerShell 5.1 would wrap it in error records).
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [switch]$Capture
    )
    $ErrorActionPreference = 'Continue'
    Write-Host ("[build.ps1] > " + $FilePath + " " + ($ArgumentList -join ' '))
    if ($Capture) {
        $output = & $FilePath @ArgumentList
        $code = $LASTEXITCODE
    }
    else {
        & $FilePath @ArgumentList | Out-Host
        $code = $LASTEXITCODE
    }
    if ($code -ne 0) {
        throw ("command failed (exit " + $code + "): " + $FilePath + " " + ($ArgumentList -join ' '))
    }
    if ($Capture) { return $output }
}

function Get-NativeExitCode {
    # Runs a native command only for its exit code: output is discarded and a non-zero exit is returned,
    # not thrown (Invoke-Native throws on every non-zero exit, so it cannot tell 1 from an error).
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @()
    )
    $ErrorActionPreference = 'Continue'
    Write-Host ("[build.ps1] > " + $FilePath + " " + ($ArgumentList -join ' '))
    & $FilePath @ArgumentList | Out-Null
    return $LASTEXITCODE
}

function Get-LineCount($Lines) {
    if ($null -eq $Lines) { return 0 }
    return @($Lines).Count
}

$savedEnv = @{}
foreach ($name in @('PYTHONDONTWRITEBYTECODE', 'PYTHONUTF8')) {
    $savedEnv[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$savedOutputEncoding = $null
try { $savedOutputEncoding = [Console]::OutputEncoding } catch { $savedOutputEncoding = $null }
$pushed = $false
$exitCode = 0

try {
    # Native tools (git, python, docker) print UTF-8; decode captured text as UTF-8.
    try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false) } catch { }

    $RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot 'docker\finalize.sh'))) {
        throw ("not the repository root (docker\finalize.sh missing): " + $RepoRoot)
    }

    # ---------------------------------------------------------------- context path guards
    if (-not $Context) {
        if (-not $env:USERPROFILE) { throw 'USERPROFILE is not set; pass -Context' }
        $Context = Join-Path $env:USERPROFILE 'nmfc-ctx'
    }
    $Context = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Context).TrimEnd('\', '/')
    if ($Context -match '[^\x20-\x7E]') {
        throw ("context path must be ASCII (old BuildKit fails on non-ASCII contexts): " + $Context)
    }
    if ((Split-Path -Leaf $Context) -notlike 'nmfc-ctx*') {
        throw ("context directory name must start with 'nmfc-ctx' (it is deleted and re-created): " + $Context)
    }
    $repoPrefix = $RepoRoot.TrimEnd('\') + '\'
    if (($Context + '\').StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw ("context must be outside the repository: " + $Context)
    }
    $ctxParent = Split-Path -Parent $Context
    if (-not $ctxParent -or -not (Test-Path -LiteralPath $ctxParent -PathType Container)) {
        throw ("parent directory of the context does not exist: " + $ctxParent)
    }

    Invoke-Native -FilePath 'git' -ArgumentList @('-C', $RepoRoot, 'rev-parse', '--is-inside-work-tree') -Capture | Out-Null

    # ---------------------------------------------------------------- 1. export
    if (Test-Path -LiteralPath $Context) {
        Write-Step ("removing old context " + $Context)
        Remove-Item -LiteralPath $Context -Recurse -Force
    }
    New-Item -ItemType Directory -Path $Context | Out-Null

    $prefix = ($Context -replace '\\', '/') + '/'
    $indexFiles = Get-LineCount (Invoke-Native -FilePath 'git' -ArgumentList @('-C', $RepoRoot, 'ls-files') -Capture)
    if ($indexFiles -gt 0) {
        Write-Step ("export: git checkout-index -a -f (" + $indexFiles + " index entries) -> " + $Context)
        Invoke-Native -FilePath 'git' -ArgumentList @('-C', $RepoRoot, 'checkout-index', '-a', '-f', ('--prefix=' + $prefix))
    }
    else {
        # Empty index: export HEAD (the same content 'git archive HEAD' would give) through a temporary
        # index file, so the real index is never touched. 'git archive | tar' is not used here because
        # Windows tar.exe cannot extract UTF-8 (Korean) path names ("Invalid empty pathname", measured).
        Write-Step 'export: the index is empty - exporting HEAD via a temporary index (git read-tree HEAD)'
        $tmpIndex = Join-Path $ctxParent ('nmfc-ctx-index-' + $PID)
        $savedGitIndex = [Environment]::GetEnvironmentVariable('GIT_INDEX_FILE', 'Process')
        try {
            if (Test-Path -LiteralPath $tmpIndex) { Remove-Item -LiteralPath $tmpIndex -Force }
            $env:GIT_INDEX_FILE = $tmpIndex
            Invoke-Native -FilePath 'git' -ArgumentList @('-C', $RepoRoot, 'read-tree', 'HEAD')
            Invoke-Native -FilePath 'git' -ArgumentList @('-C', $RepoRoot, 'checkout-index', '-a', '-f', ('--prefix=' + $prefix))
        }
        finally {
            [Environment]::SetEnvironmentVariable('GIT_INDEX_FILE', $savedGitIndex, 'Process')
            if (Test-Path -LiteralPath $tmpIndex) { Remove-Item -LiteralPath $tmpIndex -Force }
        }
    }
    # Only what is NOT in the exported index matters: unstaged edits (worktree vs index) and untracked,
    # non-ignored files. Staged entries (index vs HEAD) ARE exported, so they are not counted
    # (counting 'git status --porcelain' lines warned falsely right after a 'git add -A').
    $unstaged = Get-LineCount (Invoke-Native -FilePath 'git' -ArgumentList @('-C', $RepoRoot, '--no-optional-locks', 'diff', '--name-only') -Capture)
    $untracked = Get-LineCount (Invoke-Native -FilePath 'git' -ArgumentList @('-C', $RepoRoot, '--no-optional-locks', 'ls-files', '-o', '--exclude-standard') -Capture)
    if (($unstaged + $untracked) -gt 0) {
        Write-Warning ("[build.ps1] not in the exported tree: " + $unstaged + " unstaged change(s), " + $untracked + " untracked file(s). The image is built from the git index (HEAD if the index is empty); run 'git add -A' first if they should be included")
    }
    else {
        Write-Step 'working tree = index (no unstaged changes, no untracked files)'
    }
    # Revision label (org.opencontainers.image.revision) follows what was exported: '+index' when the exported
    # index differs from HEAD; no suffix when an empty index made it export HEAD; 'no-commit' when HEAD does not exist.
    $revision = 'no-commit'
    if ((Get-NativeExitCode -FilePath 'git' -ArgumentList @('-C', $RepoRoot, 'rev-parse', '--verify', '--quiet', 'HEAD')) -eq 0) {
        $revision = (Invoke-Native -FilePath 'git' -ArgumentList @('-C', $RepoRoot, 'rev-parse', '--short=12', 'HEAD') -Capture) -join ''
        if ($indexFiles -gt 0) {
            $cachedCode = Get-NativeExitCode -FilePath 'git' -ArgumentList @('-C', $RepoRoot, '--no-optional-locks', 'diff', '--cached', '--quiet')
            if ($cachedCode -eq 1) {
                $revision = $revision + '+index'
            }
            elseif ($cachedCode -ne 0) {
                throw ("command failed (exit " + $cachedCode + "): git diff --cached --quiet")
            }
        }
    }
    Write-Step ("revision: NMFC_REVISION=" + $revision)

    # ---------------------------------------------------------------- 2. lock
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONUTF8 = '1'
    Push-Location -LiteralPath $Context
    $pushed = $true
    Invoke-Native -FilePath $Python -ArgumentList @('-B', 'docker/nmfc_cli.py', 'manifest', '--write-lock', $Context)
    Pop-Location
    $pushed = $false

    $lockPath = Join-Path $Context 'docker\context.lock'
    $filesPath = Join-Path $Context 'docker\manifest.files'
    foreach ($p in @($lockPath, $filesPath)) {
        if (-not (Test-Path -LiteralPath $p -PathType Leaf) -or (Get-Item -LiteralPath $p).Length -eq 0) {
            throw ("manifest --write-lock did not produce " + $p)
        }
    }
    $lockBytes = [System.IO.File]::ReadAllBytes($lockPath)
    if ([Array]::IndexOf($lockBytes, [byte]13) -ge 0) {
        throw 'docker/context.lock contains CR bytes; it must be written with LF so that its sha256 is platform independent'
    }

    # ---------------------------------------------------------------- 3. hash
    $sha = (Get-FileHash -LiteralPath $lockPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $nFiles = Get-LineCount (Get-Content -LiteralPath $filesPath -Encoding UTF8)
    Write-Step ("NMFC_MANIFEST_SHA256=" + $sha + " (" + $nFiles + " files in manifest)")

    if ($SkipBuild) {
        Write-Step '-SkipBuild: stopping before docker build'
    }
    else {
        # ------------------------------------------------------------ 4. build
        $tag = 'nmfc:' + $Target
        Invoke-Native -FilePath $Docker -ArgumentList @(
            'build', '--provenance=false', '--progress=plain', '--target', $Target,
            '--build-arg', ('NMFC_MANIFEST_SHA256=' + $sha),
            '--build-arg', 'NMFC_MANIFEST_SOURCE=host-export',
            '--build-arg', ('NMFC_REVISION=' + $revision),
            '-t', $tag, $Context)

        # ------------------------------------------------------------ 5. record
        $inspectText = (Invoke-Native -FilePath $Docker -ArgumentList @('image', 'inspect', $tag) -Capture) -join "`n"
        $img = @(ConvertFrom-Json -InputObject $inspectText)[0]
        $labels = [ordered]@{}
        if ($null -ne $img.Config.Labels) {
            foreach ($prop in $img.Config.Labels.PSObject.Properties) { $labels[$prop.Name] = $prop.Value }
        }
        $wantBackend = 'cpu'
        if ($Target -eq 'gpu') { $wantBackend = 'cu128' }
        if (-not $labels.Contains('nmfc.manifest.sha256') -or $labels['nmfc.manifest.sha256'] -ne $sha) {
            throw 'image label nmfc.manifest.sha256 does not match the context lock'
        }
        if (-not $labels.Contains('nmfc.manifest.source') -or $labels['nmfc.manifest.source'] -ne 'host-export') {
            throw 'image label nmfc.manifest.source is not host-export'
        }
        if (-not $labels.Contains('nmfc.backend') -or $labels['nmfc.backend'] -ne $wantBackend) {
            throw ('image label nmfc.backend is not ' + $wantBackend)
        }
        if (-not $labels.Contains('org.opencontainers.image.revision') -or $labels['org.opencontainers.image.revision'] -ne $revision) {
            throw ('image label org.opencontainers.image.revision is not the build argument NMFC_REVISION=' + $revision)
        }
        $serverVersion = ''
        $driver = ''
        $driverStatus = ''
        try { $serverVersion = (Invoke-Native -FilePath $Docker -ArgumentList @('version', '--format', '{{.Server.Version}}') -Capture) -join '' } catch { }
        try { $driver = (Invoke-Native -FilePath $Docker -ArgumentList @('info', '--format', '{{.Driver}}') -Capture) -join '' } catch { }
        try { $driverStatus = (Invoke-Native -FilePath $Docker -ArgumentList @('info', '--format', '{{json .DriverStatus}}') -Capture) -join '' } catch { }

        $record = [ordered]@{
            target                = $Target
            tag                   = $tag
            id                    = $img.Id
            rootfs_layers         = @($img.RootFS.Layers)
            size                  = $img.Size
            labels                = $labels
            created               = $img.Created
            architecture          = $img.Architecture
            os                    = $img.Os
            manifest_sha256       = $sha
            manifest_source       = 'host-export'
            manifest_files        = $nFiles
            context               = $Context
            docker_server_version = $serverVersion
            storage_driver        = $driver
            driver_status         = $driverStatus
            recorded_at_utc       = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd'T'HH:mm:ss'Z'")
        }
        $outDir = Join-Path $RepoRoot 'out'
        if (-not (Test-Path -LiteralPath $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
        $outPath = Join-Path $outDir ('build_' + $Target + '.json')
        $json = ($record | ConvertTo-Json -Depth 8) -replace "`r`n", "`n"
        [System.IO.File]::WriteAllText($outPath, $json + "`n", (New-Object System.Text.UTF8Encoding($false)))
        Write-Step ("built " + $tag + " " + $img.Id + " (" + @($img.RootFS.Layers).Count + " layers, " + $img.Size + " bytes)")
        Write-Step ("record -> " + $outPath)
    }
}
catch {
    Write-Host ("[build.ps1] FAILED: " + $_.Exception.Message) -ForegroundColor Red
    $exitCode = 1
}
finally {
    if ($pushed) { Pop-Location }
    foreach ($name in $savedEnv.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnv[$name], 'Process')
    }
    if ($null -ne $savedOutputEncoding) {
        try { [Console]::OutputEncoding = $savedOutputEncoding } catch { }
    }
}
exit $exitCode
