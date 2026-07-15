terraform {
  required_version = ">= 1.10, < 2.0"

  # CI supplies the bucket, region, and key as a partial configuration. Keeping
  # credentials out of this block prevents them from leaking into plans.
  backend "s3" {}

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.51"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.tags
  }
}

data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  name = "${var.project}-${var.environment}"
  tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

check "isolated_workspace" {
  assert {
    condition     = terraform.workspace == var.environment
    error_message = "Select a Terraform workspace matching var.environment before planning or applying."
  }
}
