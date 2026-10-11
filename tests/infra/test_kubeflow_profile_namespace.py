import json
import os
import subprocess
from pathlib import Path

import yaml


def test_rendered_resources_use_the_gods_profile_namespace():
    root = Path(__file__).resolve().parents[2]
    rendered_path = os.environ.get('GODS_KUBEFLOW_RENDERED_PATH')
    if rendered_path:
        rendered = Path(rendered_path).read_text()
    else:
        rendered = subprocess.run(
            ['kubectl', 'kustomize', str(root / 'infra/kubeflow')],
            capture_output=True, text=True, check=True, timeout=180,
        ).stdout
    loader = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)
    resources = [item for item in yaml.load_all(rendered, Loader=loader) if item]
    assert any(item['kind'] == 'Profile' and item['metadata']['name'] == 'gods-mlops'
               for item in resources)
    assert 'kubeflow-user-example-com' not in json.dumps(resources)
    registry = next(item for item in resources
                    if item['kind'] == 'Deployment'
                    and item['metadata']['name'] == 'model-registry-deployment')
    assert registry['metadata']['namespace'] == 'gods-mlops'
