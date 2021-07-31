import time
import unittest

from mprime.mprime import MPrime, Statuses


class TestIntegration(unittest.TestCase):
    def test_process_works(self):
        num_threads = 2
        config = dict(
            TortureThreads=num_threads,
            TortureTime=0.2,
            TortureMem=4,
            MaxTortureFFT=128,
            MinTortureFFT=2,
        )

        mprime = MPrime(config)
        uncaught_output = []
        mprime.handlers["on_uncaught_output"] = lambda o: uncaught_output.append(o)

        mprime.launch_mprime()
        time.sleep(30)
        self.assertEqual(mprime.status, Statuses.running)
        self.assertEqual(len(mprime.workers), num_threads)
        pre_stop_time = time.time()
        mprime.stop()

        # ensure the program shutsdown in a reasonable time
        self.assertLess(time.time() - pre_stop_time, 2)
        self.assertEqual(mprime.status, Statuses.stopped)

        self.assertEqual(len(uncaught_output), 0, uncaught_output)
