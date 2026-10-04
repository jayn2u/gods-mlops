resource "helm_release" "nvidia_device_plugin" {
  name             = "nvidia-device-plugin"
  repository       = "https://nvidia.github.io/k8s-device-plugin"
  chart            = "nvidia-device-plugin"
  version          = local.versions_lock.platform.nvidia_device_plugin
  namespace        = "nvidia-device-plugin"
  create_namespace = true
  wait             = true
  timeout          = 600

  values = [
    yamlencode({
      runtimeClassName   = "nvidia"
      nodeSelector       = { "kubernetes.io/hostname" = "ubuntu" }
      nfd                = { enabled = false }
      gfd                = { enabled = false }
      deviceListStrategy = "envvar"
      deviceIDStrategy   = "uuid"
      failOnInitError    = true
      nvidiaDriverRoot   = "/"
    })
  ]

  depends_on = [terraform_data.preflight_gate]
}
