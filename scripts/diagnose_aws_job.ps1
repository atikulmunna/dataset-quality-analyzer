param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[A-Za-z0-9_-]{1,64}$")]
    [string]$JobId,
    [Parameter(Mandatory = $true)]
    [string]$TableName,
    [Parameter(Mandatory = $true)]
    [string]$QueueArn,
    [Parameter(Mandatory = $true)]
    [string]$WorkerLogGroup,
    [Parameter(Mandatory = $true)]
    [string]$MonitorLogGroup,
    [string]$Region = "us-east-1"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Durable job record ==="
aws dynamodb get-item `
    --region $Region `
    --table-name $TableName `
    --consistent-read `
    --key "pk={S=JOB#$JobId}" `
    --output json
if ($LASTEXITCODE -ne 0) { throw "Unable to read the job record." }

$jobName = "dqa-$($JobId.Substring(0, [Math]::Min(32, $JobId.Length)))"
$batchIds = aws batch list-jobs `
    --region $Region `
    --job-queue $QueueArn `
    --filters "name=JOB_NAME,values=$jobName" `
    --query "jobSummaryList[].jobId" `
    --output text
if ($LASTEXITCODE -ne 0) { throw "Unable to search AWS Batch jobs." }

Write-Host "=== AWS Batch attempts ==="
if ($batchIds) {
    aws batch describe-jobs --region $Region --jobs ($batchIds -split "\s+") --output json
    if ($LASTEXITCODE -ne 0) { throw "Unable to describe AWS Batch attempts." }
} else {
    Write-Host "No retained Batch attempt matched $jobName."
}

$filter = "{ $.job_id = `"$JobId`" }"
foreach ($group in @($WorkerLogGroup, $MonitorLogGroup)) {
    Write-Host "=== Structured events: $group ==="
    aws logs filter-log-events `
        --region $Region `
        --log-group-name $group `
        --filter-pattern $filter `
        --query "events[].{timestamp:timestamp,message:message}" `
        --output json
    if ($LASTEXITCODE -ne 0) { throw "Unable to search $group." }
}
