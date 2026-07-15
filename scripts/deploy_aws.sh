#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <dev|prod> <40-character-commit-sha>" >&2
  exit 2
fi

environment="$1"
release_sha="$2"

if [[ "$environment" != "dev" && "$environment" != "prod" ]]; then
  echo "environment must be dev or prod" >&2
  exit 2
fi
if [[ ! "$release_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "release SHA must be 40 lowercase hexadecimal characters" >&2
  exit 2
fi
: "${AWS_REGION:?AWS_REGION is required}"
: "${TF_STATE_BUCKET:?TF_STATE_BUCKET is required}"
: "${ALERT_EMAIL:?ALERT_EMAIL is required}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module="$root/infra/terraform"
image_archive="$root/.ci/dqa-worker.tar.gz"

python "$root/scripts/build_lambda.py"

terraform -chdir="$module" init -input=false -reconfigure \
  -backend-config="bucket=$TF_STATE_BUCKET" \
  -backend-config="key=dqa/terraform.tfstate" \
  -backend-config="region=$AWS_REGION" \
  -backend-config="encrypt=true" \
  -backend-config="use_lockfile=true" \
  -backend-config="workspace_key_prefix=dqa-workspaces"

terraform -chdir="$module" workspace select "$environment" \
  || terraform -chdir="$module" workspace new "$environment"

terraform -chdir="$module" apply -input=false -auto-approve \
  -var-file="environments/$environment.tfvars" \
  -var="aws_region=$AWS_REGION" \
  -var="alert_email=$ALERT_EMAIL" \
  -var="worker_image_tag=$release_sha" \
  -var="admission_enabled=false"

repository_url="$(terraform -chdir="$module" output -raw worker_repository_url)"
repository_name="${repository_url##*/}"

if aws ecr describe-images \
  --region "$AWS_REGION" \
  --repository-name "$repository_name" \
  --image-ids "imageTag=$release_sha" >/dev/null 2>&1; then
  echo "immutable worker image $release_sha already exists; reusing it"
else
  if [[ ! -f "$image_archive" ]]; then
    echo "worker image artifact is missing: $image_archive" >&2
    exit 1
  fi
  gunzip -c "$image_archive" | docker load
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "${repository_url%%/*}"
  docker tag dqa-worker:local "$repository_url:$release_sha"
  docker push "$repository_url:$release_sha"
fi

api_url="$(terraform -chdir="$module" output -raw api_url)"
ui_url="$(terraform -chdir="$module" output -raw ui_url)"
ui_bucket="$(terraform -chdir="$module" output -raw ui_bucket)"
ui_distribution_id="$(terraform -chdir="$module" output -raw ui_distribution_id)"
cognito_domain="$(terraform -chdir="$module" output -raw cognito_hosted_ui_domain)"
cognito_client_id="$(terraform -chdir="$module" output -raw cognito_client_id)"

python "$root/scripts/build_web.py" \
  --mode live \
  --api-base-url "$api_url" \
  --cognito-domain "$cognito_domain" \
  --cognito-client-id "$cognito_client_id" \
  --cognito-redirect-uri "$ui_url/"

aws s3 sync "$root/dist/web" "s3://$ui_bucket" \
  --delete \
  --cache-control "public,max-age=3600"
aws s3 cp "$root/dist/web/index.html" "s3://$ui_bucket/index.html" \
  --content-type "text/html" \
  --cache-control "no-cache"
aws s3 cp "$root/dist/web/config.js" "s3://$ui_bucket/config.js" \
  --content-type "application/javascript" \
  --cache-control "no-store"
invalidation_id="$(aws cloudfront create-invalidation \
  --distribution-id "$ui_distribution_id" \
  --paths "/*" \
  --query 'Invalidation.Id' \
  --output text)"
aws cloudfront wait invalidation-completed \
  --distribution-id "$ui_distribution_id" \
  --id "$invalidation_id"

health_body="$(mktemp)"
ui_body="$(mktemp)"
trap 'rm -f "$health_body" "$ui_body"' EXIT

health_status="$(curl --silent --show-error --retry 5 --retry-all-errors \
  --retry-delay 2 --output "$health_body" --write-out '%{http_code}' \
  "$api_url/health")"
if [[ "$health_status" != "200" ]] || ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"' "$health_body"; then
  echo "health smoke failed with HTTP $health_status" >&2
  cat "$health_body" >&2
  exit 1
fi

ui_status="$(curl --silent --show-error --retry 8 --retry-all-errors \
  --retry-delay 3 --output "$ui_body" --write-out '%{http_code}' \
  "$ui_url/")"
if [[ "$ui_status" != "200" ]] || ! grep -q 'Dataset Quality Analyzer' "$ui_body"; then
  echo "UI smoke failed with HTTP $ui_status" >&2
  exit 1
fi

auth_status="$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
  --request POST --header 'content-type: application/json' --data '{}' "$api_url/jobs")"
if [[ "$auth_status" != "401" ]]; then
  echo "authentication smoke failed: unauthenticated POST /jobs returned $auth_status" >&2
  exit 1
fi

admission="$(terraform -chdir="$module" output -raw admission_enabled)"
if [[ "$admission" != "false" ]]; then
  echo "deployment unexpectedly opened job admission" >&2
  exit 1
fi

echo "$environment deployment passed UI, health, authentication, and fail-closed admission smoke checks"
echo "UI URL: $ui_url/"
