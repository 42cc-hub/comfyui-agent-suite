# win-balloon.ps1 - Windows 托盘气泡（系统通知）原语
# 用途：人不在界面时，用系统通知把他叫回来。无需注册 AUMID、最稳。
# 用法：powershell -NoProfile -ExecutionPolicy Bypass -File win-balloon.ps1 `
#         -Title "DSH Agent" -Message "需要你拍板" -Duration 8000
param(
  [string]$Title = 'DSH Agent',
  [string]$Message = '',
  [int]$Duration = 8000
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$n = New-Object System.Windows.Forms.NotifyIcon
$n.Icon = [System.Drawing.SystemIcons]::Information
$n.Visible = $true
$n.BalloonTipTitle = $Title
$n.BalloonTipText = $Message
$n.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Info
$n.ShowBalloonTip($Duration)

# 保持 NotifyIcon 存活到气泡显示完
Start-Sleep -Seconds ([math]::Ceiling($Duration / 1000.0) + 1)
$n.Dispose()
