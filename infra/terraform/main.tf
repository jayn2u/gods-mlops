terraform {
  required_version = ">= 1.7.0"

  required_providers {
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.14"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.31"
    }
  }
}

provider "kubernetes" {
  config_path    = var.kubeconfig_path
  config_context = var.kube_context
}

provider "helm" {
  kubernetes {
    config_path    = var.kubeconfig_path
    config_context = var.kube_context
  }
}

locals {
  versions_lock = yamldecode(file(var.versions_lock_path))
  image_locks = {
    for image in local.versions_lock.images : image.name => image
  }
  preflight_report = fileexists(var.preflight_report_path) ? jsondecode(file(var.preflight_report_path)) : {}
  preflight_nodes  = try(local.preflight_report.nodes, [])
  preflight_by_node = {
    for node in local.preflight_nodes : try(node.node, "unknown") => node
  }
  preflight_is_ready = try(
    local.preflight_report.schema_version == 1 &&
    local.preflight_report.status == "ready" &&
    length(local.preflight_nodes) == 2 &&
    alltrue([
      for node_name in ["ubuntu", "vis-lab"] :
      local.preflight_by_node[node_name].status == "ready" &&
      tobool(tostring(local.preflight_by_node[node_name].permissions.sudo_noninteractive)) &&
      tobool(tostring(local.preflight_by_node[node_name].storage.owner_marker_valid))
    ]),
    false
  )
  storage_image         = local.image_locks["object-storage"]
  storage_image_digest  = "${local.storage_image.source}@${local.storage_image.digest}"
  storage_chart_version = "${local.versions_lock.storage.version}.0"
}

resource "terraform_data" "preflight_gate" {
  input = {
    preflight_report_sha256 = fileexists(var.preflight_report_path) ? filesha256(var.preflight_report_path) : "missing"
    versions_lock_sha256    = filesha256(var.versions_lock_path)
    ready                   = local.preflight_is_ready
  }

  lifecycle {
    precondition {
      condition     = local.preflight_is_ready
      error_message = "Terraform is blocked until infra/ansible/preflight.yml writes a ready, ownership-checked report for both nodes."
    }

    precondition {
      condition = try(
        local.versions_lock.platform.k3s == "v1.36.2+k3s1" &&
        local.versions_lock.platform.kubeflow.version == "26.03.1" &&
        local.versions_lock.platform.nvidia_device_plugin == "0.19.3" &&
        local.storage_image.tag == local.versions_lock.storage.version &&
        local.storage_image.digest == "sha256:ce9e796f1fe6f06968f4c04bdaf8f678dad9c8acdfef3d244133d71bfa6bf882",
        false
      )
      error_message = "The platform or storage image versions differ from the reviewed immutable lock."
    }
  }
}

data "kubernetes_namespace_v1" "gods_mlops" {
  metadata {
    name = var.kubeflow_namespace
  }

  depends_on = [terraform_data.preflight_gate]
}

output "kubeflow_namespace" {
  description = "Kubeflow Profile namespace consumed by the Gods workloads."
  value       = data.kubernetes_namespace_v1.gods_mlops.metadata[0].name
}

output "retained_claim_names" {
  description = "Kustomize-owned static claims; do not recreate them through Terraform."
  value = [
    "gods-mlops-objects",
    "gods-mlops-cache",
    "gods-mlops-metadata",
    "gods-mlops-spool",
    "gods-mlops-ingestion-postgres",
  ]
}

output "object_storage_release" {
  description = "The SeaweedFS S3-compatible Helm release installed in the Kubeflow namespace."
  value       = helm_release.seaweedfs.name
}

output "object_storage_image" {
  description = "Digest-pinned SeaweedFS image selected from the Task 1 lock."
  value       = local.storage_image_digest
}

output "nvidia_device_plugin_chart_version" {
  description = "Locked NVIDIA Kubernetes device-plugin Helm chart version."
  value       = helm_release.nvidia_device_plugin.version
}
