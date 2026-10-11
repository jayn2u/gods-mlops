"""Opt-in real image regression for the non-root SSH observer."""
import os
import subprocess
import unittest


@unittest.skipUnless(os.environ.get("GODS_OBSERVER_TEST_IMAGE"), "requires built image")
class ObserverImageUserTest(unittest.TestCase):
    def test_runtime_uid_has_passwd_entry(self):
        result = subprocess.run(
            ["docker", "run", "--rm", "--network=none", "--user", "10001:10001",
             "--entrypoint", "python", os.environ["GODS_OBSERVER_TEST_IMAGE"],
             "-c", "import os,pwd; assert pwd.getpwuid(os.getuid()).pw_uid == 10001"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
