import os
from pathlib import Path

import yaml


def test_trainer_webhook_policy_selects_the_actual_controller():
    rendered = Path(os.environ['GODS_KUBEFLOW_RENDERED_PATH']).read_text()
    docs = list(yaml.load_all(rendered, Loader=yaml.CSafeLoader))
    policy = next(d for d in docs if d and d['kind'] == 'NetworkPolicy'
                  and d['metadata']['name'] == 'trainer-webhook')
    controller = next(d for d in docs if d and d['kind'] == 'Deployment'
                      and d['metadata']['name'] == 'kubeflow-trainer-controller-manager')
    labels = controller['spec']['template']['metadata']['labels']
    selector = policy['spec']['podSelector']['matchLabels']
    assert selector and all(labels.get(k) == v for k, v in selector.items())
    assert policy['spec']['ingress'] == [{'ports': [{'port': 9443, 'protocol': 'TCP'}]}]
