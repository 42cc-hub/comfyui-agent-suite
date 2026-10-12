# dsh-ps / ps_bridge.ps1
# 常驻 PowerShell 会话，持有 Photoshop COM 会话，按 NDJSON 协议收发命令。
# 协议（stdin/stdout 各一行为一条 JSON）：
#   入: {"id":N,"script":"<ExtendScript>"}  出: {"id":N,"r":"<result>"} 或 {"id":N,"e":"<err>"}
#   入: {"id":N,"type":"quit"}            出: {"id":N,"r":"quit"}
$ErrorActionPreference = 'Stop'
$ps = New-Object -ComObject Photoshop.Application
[Console]::Out.WriteLine((ConvertTo-Json -Compress @{ ready = $true; version = $ps.Version }))
[Console]::Out.Flush()

while ($true) {
  $line = [Console]::In.ReadLine()
  if ($null -eq $line) { break }
  if ([string]::IsNullOrWhiteSpace($line)) { continue }
  $req = $null
  try {
    $req = $line | ConvertFrom-Json
    $id = if ($req.id -ne $null) { $req.id } else { 0 }
    if ($req.type -eq 'quit') {
      $ps.Quit()
      [Console]::Out.WriteLine((ConvertTo-Json -Compress @{ id = $id; r = 'quit' }))
      [Console]::Out.Flush()
      break
    }
    $r = $ps.DoJavaScript($req.script)
    [Console]::Out.WriteLine((ConvertTo-Json -Compress @{ id = $id; r = $r }))
    [Console]::Out.Flush()
  } catch {
    $eid = 0; if ($req -ne $null -and $req.id -ne $null) { $eid = $req.id }
    [Console]::Out.WriteLine((ConvertTo-Json -Compress @{ id = $eid; e = $_.Exception.Message }))
    [Console]::Out.Flush()
  }
}
