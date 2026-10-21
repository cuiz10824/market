param([switch]$Remove)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$hash = [System.Security.Cryptography.SHA256]::Create()
$id = [BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($root.ToLowerInvariant()))).Replace('-','').Substring(0,8)
$taskName = 'AShareMarketWeeklyAgent-' + $id
if ($Remove) {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($task) {
        Stop-ScheduledTask -TaskName $taskName
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    }
    Write-Host 'Background update task removed.'
    exit 0
}
$python = Join-Path $root '.venv-daily\Scripts\pythonw.exe'
if (!(Test-Path $python)) { throw 'Run the dashboard launcher once to install Python dependencies.' }
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $python -Argument ('"' + (Join-Path $PSScriptRoot 'server.py') + '" --no-browser --background') -WorkingDirectory $root
$triggers = @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At '09:00'),
    (New-ScheduledTaskTrigger -AtLogOn -User $user)
)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $triggers -Settings $settings -Principal $principal -Description 'Collect completed A-share index candles, verify weekly coverage and refresh saved-model forecasts.' -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Host 'Installed and started. The updater runs while this Windows user is signed in.'
Write-Host 'Weekly data checks use Asia/Shanghai time; startup catches missed weeks.'
Write-Host ('Task: ' + $taskName)
