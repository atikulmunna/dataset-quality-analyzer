param(
    [string]$Region = "us-east-1",
    [string]$StackName = "dqa-deployment-bootstrap"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$template = Join-Path $root "infra/bootstrap/github-oidc.yaml"
$providerArn = aws iam list-open-id-connect-providers `
    --query "OpenIDConnectProviderList[?contains(Arn, 'token.actions.githubusercontent.com')].Arn | [0]" `
    --output text
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect existing AWS OIDC providers."
}

$arguments = @(
    "cloudformation", "deploy",
    "--region", $Region,
    "--stack-name", $StackName,
    "--template-file", $template,
    "--capabilities", "CAPABILITY_NAMED_IAM",
    "--no-fail-on-empty-changeset"
)
if ($providerArn -and $providerArn -ne "None") {
    $arguments += @(
        "--parameter-overrides",
        "ExistingGitHubOidcProviderArn=$providerArn"
    )
    Write-Host "Reusing account-level GitHub OIDC provider $providerArn"
}

aws @arguments
if ($LASTEXITCODE -ne 0) {
    throw "AWS deployment bootstrap failed."
}

aws cloudformation describe-stacks `
    --region $Region `
    --stack-name $StackName `
    --query "Stacks[0].Outputs[].{Name:OutputKey,Value:OutputValue}" `
    --output table
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read bootstrap outputs."
}
