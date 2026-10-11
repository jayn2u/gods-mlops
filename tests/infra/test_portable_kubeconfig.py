from pathlib import Path
import yaml


def test_controller_fetches_portable_k3s_kubeconfig_without_logging_credentials():
    root = Path(__file__).resolve().parents[2]
    plays = yaml.safe_load((root / 'infra/ansible/site.yml').read_text())
    task = next(t for p in plays for t in p.get('tasks', [])
                if t.get('name') == 'Fetch the Gods control-plane kubeconfig to the controller')
    assert task['ansible.builtin.fetch']['src'] == '/etc/rancher/k3s/k3s.yaml'
    assert task['no_log'] is True
