import os
import unittest
from mprime.mprime import MPrime, Statuses

this_dir = os.path.dirname(__file__)


class TestParsing(unittest.TestCase):
    class FilePaths:
        class FullCapture:
            dummy_run = this_dir + "/terminal_captures/dummy_run.out"

        class PartialCapture:
            startup = this_dir + "/terminal_captures/startup.out"
            tests_running = this_dir + "/terminal_captures/tests_running.out"
            tests_completed = this_dir + "/terminal_captures/tests_completed.out"
            process_terminated = this_dir + "/terminal_captures/process_terminated.out"

    def setUp(self) -> None:
        self.mprime = MPrime()
        self.uncaught_output = []
        self.mprime.handlers[
            "on_uncaught_output"
        ] = lambda o: self.uncaught_output.append(o)

    def load_terminal_capture(self, path):
        with open(path, "rb") as f:
            self.mprime._consume_mprime_output(f)

    def assert_all_workers_registered(self):
        # there were 16 threads in the recorded run - there should be 16 workers registered.
        self.assertEqual(16, len(self.mprime.workers))

    def test_startup_workers(self):
        self.load_terminal_capture(self.FilePaths.PartialCapture.startup)

        self.assert_all_workers_registered()
        self.assertEqual(self.mprime.status, Statuses.running)
        for worker in self.mprime.workers.values():
            self.assertEqual(worker.status, Statuses.starting)
            self.assertEqual(len(worker.tests), 0)

    def test_startup_running_first_task(self):
        self.load_terminal_capture(self.FilePaths.PartialCapture.startup)
        self.load_terminal_capture(self.FilePaths.PartialCapture.tests_running)

        self.assert_all_workers_registered()
        self.assertEqual(self.mprime.status, Statuses.running)

        for worker in self.mprime.workers.values():
            self.assertEqual(worker.status, Statuses.running)
            self.assertEqual(len(worker.tests), 1)

    def test_parsing_mprime_full_output(self):
        self.load_terminal_capture(self.FilePaths.FullCapture.dummy_run)

        self.assert_all_workers_registered()

        # ensure the object fully represents the fact that execution has ended
        self.assertFalse(self.mprime.running)
        self.assertEqual(self.mprime.status, Statuses.stopping)

        self.mprime.stop()

        self.assertEqual(self.mprime.status, Statuses.stopped)

        # ensure all output was filtered
        self.assertEqual(0, len(self.uncaught_output), self.uncaught_output)

    def test_dict_to_ini(self):
        generated_ini_string = MPrime.dict_to_ini(
            dict(a=1, b=2, c=None, d="123123", e="False", abcde=False)
        )

        expected_ini_string = (
            "a=1",
            "b=2",
            "d=123123",
            "e=False",
            "abcde=False",
        )

        for line in expected_ini_string:
            self.assertIn(line, generated_ini_string)
