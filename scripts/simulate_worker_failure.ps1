param(
    [Parameter(Mandatory = $true)]
    [string]$MonitorFunctionName,
    [Parameter(Mandatory = $true)]
    [string]$QueueArn,
    [string]$Environment = "dev",
    [string]$Region = "us-east-1"
)

$ErrorActionPreference = "Stop"
$now = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
$jobId = "alarm-test-$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
$event = @{
    "detail-type" = "Batch Job State Change"
    source = "aws.batch"
    detail = @{
        jobQueue = $QueueArn
        jobId = "synthetic-$jobId"
        jobName = "dqa-$jobId"
        status = "FAILED"
        startedAt = $now - 1000
        stoppedAt = $now
        parameters = @{ job_id = $jobId }
    }
}
$payload = Join-Path ([System.IO.Path]::GetTempPath()) "$jobId.json"
$response = Join-Path ([System.IO.Path]::GetTempPath()) "$jobId-response.json"
try {
    $event | ConvertTo-Json -Depth 6 -Compress | Set-Content -LiteralPath $payload -Encoding utf8NoBOM
    aws lambda invoke `
        --region $Region `
        --function-name $MonitorFunctionName `
        --cli-binary-format raw-in-base64-out `
        --payload "fileb://$payload" `
        $response
    if ($LASTEXITCODE -ne 0) { throw "Synthetic failure invocation failed." }
    Get-Content -LiteralPath $response
    Write-Host "Published synthetic failure for $jobId. The $Environment job-failure alarm evaluates within five minutes."
} finally {
    Remove-Item -LiteralPath $payload, $response -Force -ErrorAction SilentlyContinue
}
