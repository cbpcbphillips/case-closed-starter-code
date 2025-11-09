# scripts/launch.ps1
param(
  [int]$P1Port = 5008,
  [int]$P2Port = 5009,
  [string]$P1Name = "AgentA",
  [string]$P2Name = "AgentB",
  [int]$Games = 500,
  [int]$Workers = 8
)

$ErrorActionPreference = "Stop"
$python = "$($env:VIRTUAL_ENV)\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

# Start agents in background tabs
$env:PORT = "$P1Port"; $env:AGENT_NAME = $P1Name; $env:AGENT_VARIANT = "A"; $env:LOG_FILE = "logs/$P1Name.jsonl"
Start-Process -FilePath $python -ArgumentList "agent.py" -WindowStyle Minimized

$env:PORT = "$P2Port"; $env:AGENT_NAME = $P2Name; $env:AGENT_VARIANT = "B"; $env:LOG_FILE = "logs/$P2Name.jsonl"
Start-Process -FilePath $python -ArgumentList "agent.py" -WindowStyle Minimized

Write-Host "Waiting 2s for agents to boot..."
Start-Sleep -Seconds 2

# Run parallel self-play
python tools/selfplay.py --p1 "http://127.0.0.1:$P1Port" --p2 "http://127.0.0.1:$P2Port" -n $Games -w $Workers --out tools/out

# Update Elo
python tools/elo.py --p1_name $P1Name --p2_name $P2Name --k 24
Write-Host "Done. See tools/out for results."
