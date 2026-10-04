variable "kubeconfig_path" {
  description = "Private kubeconfig written by the Gods-only Ansible K3s setup."
  type        = string
  default     = "../ansible/generated/kubeconfig"
}

variable "kube_context" {
  description = "Optional context in the private kubeconfig; null uses current-context."
  type        = string
  default     = null
  nullable    = true
}

variable "preflight_report_path" {
  description = "Read-only Ansible node, permission, ownership, and preservation report."
  type        = string
  default     = "../ansible/artifacts/preflight.json"
}

variable "versions_lock_path" {
  description = "Task 1 immutable platform and OCI image version lock."
  type        = string
  default     = "../versions.lock.yaml"
}

variable "kubeflow_namespace" {
  description = "Kubeflow Profile namespace created by the Gods Kustomize render."
  type        = string
  default     = "gods-mlops"

  validation {
    condition     = can(regex("^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", var.kubeflow_namespace))
    error_message = "kubeflow_namespace must be a valid lowercase Kubernetes namespace."
  }
}

variable "s3_admin_access_key" {
  description = "Operator-managed SeaweedFS S3 administrator access key; provide from a private tfvars file."
  type        = string
  sensitive   = true
  nullable    = false

  validation {
    condition     = length(var.s3_admin_access_key) >= 16 && !startswith(var.s3_admin_access_key, "-")
    error_message = "The S3 administrator access key must contain at least 16 characters and cannot begin with a dash."
  }
}

variable "s3_admin_secret_key" {
  description = "Operator-managed SeaweedFS S3 administrator secret key; provide from a private tfvars file."
  type        = string
  sensitive   = true
  nullable    = false

  validation {
    condition     = length(var.s3_admin_secret_key) >= 32 && !startswith(var.s3_admin_secret_key, "-")
    error_message = "The S3 administrator secret key must contain at least 32 characters and cannot begin with a dash."
  }
}

variable "s3_read_access_key" {
  description = "Non-admin read-only SeaweedFS S3 access key for internal consumers; provide privately."
  type        = string
  sensitive   = true
  nullable    = false

  validation {
    condition     = length(var.s3_read_access_key) >= 16 && !startswith(var.s3_read_access_key, "-")
    error_message = "The S3 read access key must contain at least 16 characters and cannot begin with a dash."
  }
}

variable "s3_read_secret_key" {
  description = "Non-admin read-only SeaweedFS S3 secret key for internal consumers; provide privately."
  type        = string
  sensitive   = true
  nullable    = false

  validation {
    condition     = length(var.s3_read_secret_key) >= 32 && !startswith(var.s3_read_secret_key, "-")
    error_message = "The S3 read secret key must contain at least 32 characters and cannot begin with a dash."
  }
}
