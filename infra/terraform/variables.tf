variable "project" {
  type        = string
  description = "Short resource-name prefix."
  default     = "dqa"
}

variable "environment" {
  type        = string
  description = "Isolated deployment environment and Terraform workspace name."

  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be dev or prod."
  }
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "alert_email" {
  type        = string
  description = "Address that confirms and receives budget alerts."
  default     = "atikul.munna@northsouth.edu"
}

variable "lambda_zip_path" {
  type        = string
  description = "Path to the deterministic Lambda ZIP, relative to this module."
  default     = "../../dist/lambda/dqa-api.zip"
}

variable "worker_image_tag" {
  type        = string
  description = "Immutable worker image tag; CI will replace bootstrap with a source digest tag."
  default     = "bootstrap"
}

variable "admission_enabled" {
  type        = bool
  description = "Fail-closed audit admission switch. Enable only after the worker image is published."
  default     = false
}

variable "force_destroy_buckets" {
  type        = bool
  description = "Allow Terraform destroy to delete non-empty alpha buckets."
  default     = false
}

