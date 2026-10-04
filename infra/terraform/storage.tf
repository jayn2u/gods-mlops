resource "helm_release" "seaweedfs" {
  name             = "gods-mlops-storage"
  repository       = "https://seaweedfs.github.io/seaweedfs/helm"
  chart            = "seaweedfs"
  version          = local.storage_chart_version
  namespace        = data.kubernetes_namespace_v1.gods_mlops.metadata[0].name
  create_namespace = false
  wait             = true
  atomic           = true
  timeout          = 600

  values = [
    yamlencode({
      image = {
        registry   = "ghcr.io"
        repository = "chrislusf/seaweedfs"
        tag        = local.storage_image.tag
      }
      master = {
        replicas      = 1
        imageOverride = local.storage_image_digest
        data = {
          type      = "existingClaim"
          claimName = "gods-mlops-metadata"
        }
        ingress = { enabled = false }
      }
      volume = {
        replicas      = 1
        imageOverride = local.storage_image_digest
        dataDirs = [
          {
            name       = "data"
            type       = "existingClaim"
            claimName  = "gods-mlops-objects"
            maxVolumes = 0
          }
        ]
        ingress = { enabled = false }
      }
      filer = {
        replicas      = 1
        imageOverride = local.storage_image_digest
        data = {
          type      = "existingClaim"
          claimName = "gods-mlops-metadata"
        }
        ingress = { enabled = false }
        s3 = {
          enabled    = true
          enableAuth = true
        }
      }
      s3 = {
        credentials = {
          admin = {
            accessKey = var.s3_admin_access_key
            secretKey = var.s3_admin_secret_key
          }
          read = {
            accessKey = var.s3_read_access_key
            secretKey = var.s3_read_secret_key
          }
        }
        createBuckets = [
          { name = "gods-samples", anonymousRead = false },
          { name = "gods-datasets", anonymousRead = false },
          { name = "gods-models", anonymousRead = false },
          { name = "gods-checkpoints", anonymousRead = false },
        ]
      }
      resizeHook    = { enabled = false }
      admin         = { enabled = false }
      ingress       = { enabled = false }
      networkPolicy = { enabled = false }
    })
  ]

  depends_on = [terraform_data.preflight_gate]
}
