import os
import time
from enum import Enum
from subprocess import Popen, PIPE
from threading import Thread


default_prime_config = dict(
    StressTester=1,
    UsePrimenet=0,
    MinTortureFFT=8,
    MaxTortureFFT=4098,
    TortureMem=100,
    TortureTime=3
)


Statuses = Enum('STATUSES', 'stopped starting running passed failed stopping')


class Test:
    def __init__(self, test: int, num_iterations: int, thing: str, using: str,
                 length: str, passes: list = None, clm=None):
        self.status = Statuses.running
        self.test = test
        self.num_iterations = num_iterations
        self.thing = thing
        self.using = using
        self.length = length
        self.passes = passes or []
        self.clm = clm
        self.status_reason = None


class Worker:
    """
    tests:
        Test 1, 52000 Lucas-Lehmer iterations of M6225921 using FMA3 FFT length 320K, Pass1=320, Pass2=1K, clm=1.
        Self-test 320K passed!
        Test 1, 3200000 Lucas-Lehmer iterations of M172031 using FMA3 FFT length 8K, Pass1=128, Pass2=64, clm=2.
        Self-test 8K passed!
        Test 1, 44000 Lucas-Lehmer iterations of M7471105 using FMA3 FFT length 384K, Pass1=384, Pass2=1K, clm=1.
        Self-test 384K passed!
        Test 1, 1800000 Lucas-Lehmer iterations of M250519 using FMA3 FFT length 12K, Pass1=256, Pass2=48, clm=1.
        Self-test 12K passed!
        Test 1, 36000 Lucas-Lehmer iterations of M8716289 using FMA3 FFT length 448K, Pass1=448, Pass2=1K, clm=2.
        Self-test 448K passed!
        Test 1, 1400000 Lucas-Lehmer iterations of M339487 using FMA3 FFT length 16K.
        Self-test 16K passed!
        Test 1, 31000 Lucas-Lehmer iterations of M9961473 using FMA3 FFT length 512K, Pass1=512, Pass2=1K, clm=1.
        Self-test 512K passed!
        Test 1, 1100000 Lucas-Lehmer iterations of M420217 using FMA3 FFT length 20K.

    errors:
        FATAL ERROR: Rounding was 1.047032155e+28, expected less than 0.4
        Hardware failure detected, consult stress.txt file.
        Torture Test completed 1 tests in 10 minutes - 1 errors, 0 warnings.
        Worker stopped.
    """
    def __init__(self, number: int):
        self.number = number
        self.tests = []  # type: [Test, ...]
        self.status = None  # type: Statuses
        self.status_reason = None
        self.summary = None

    def add_test(self, chunked_line: str):
        """
        parse a line of output representing a test start and add it to the workers list of tests
        """
        test, num_iterations, _, _, _, thing, _, using_a, using_b, _, length, *args = [i.strip(',') for i in chunked_line]
        if args:
            clm = args.pop(-1)
        test_obj_params = [
            test, num_iterations, thing, '{} {}'.format(using_a, using_b), length
        ] + [[args], clm] if args else []
        self.tests.append(Test(*test_obj_params))


class MPrime:
    executable = 'mprime'

    handlers = dict(
        on_worker_pass=lambda w: w,
        on_worker_fail=lambda w: w,
        on_worker_start=lambda w: w,
        on_worker_add=lambda w: w,
        on_test_complete=lambda w: w
    )

    def __init__(self, prime_config: dict = None, working_directory: str = '/tmp/mprime'):
        self.mprime = None  # type: Popen
        self.output_thread = None
        self.start_time = None
        self.stop_time = None
        self.status = Statuses.stopped
        self.workers = {}

        self.config = default_prime_config
        if prime_config is not None:
            self.config.update(prime_config)

        self.working_directory = working_directory
        if not os.path.isdir(self.working_directory):
            os.mkdir(self.working_directory)
        os.chdir(self.working_directory)

    def launch_mprime(self):
        self._write_prime_txt()
        self.mprime = Popen([self.executable, '-t'], stdout=PIPE, stderr=PIPE, stdin=PIPE)
        self.start_time = time.time()

        self.output_thread = Thread(target=self._consume_mprime_output, args=(self.mprime.stdout, ))
        self.output_thread.daemon = True  # thread dies with the program
        self.output_thread.start()

    def statusline(self):
        output_format = 'Status: {status}, Workers: ({workers_summary}), Time elapsed: {time_elapsed}'
        workers_summary = '#:{num_workers}, F:{failed_workers}'.format(
            num_workers=len(self.workers),
            failed_workers=len([w for w in self.workers.values() if w.status == Statuses.running])
        )
        return output_format.format(status=self.status, workers_summary=workers_summary, time_elapsed=self.time_elapsed)

    def __del__(self):
        """
        Might get invoked... might not...
        """
        self.stop()

    @property
    def time_elapsed(self):
        if not self.start_time:
            return None
        return (self.stop_time or time.time()) - self.start_time

    def stop(self):
        """
        kill subprocess
        """
        self.mprime.kill()
        while self.running:
            time.sleep(0.1)

    def wait_for_completion(self):
        """
        block until subprocess has finished
        """
        while self.running:
            time.sleep(10)

    @property
    def running(self) -> bool:
        """
        True if subprocess running, False otherwise
        """
        if self.mprime is None:
            return False
        return self.mprime.poll() is None

    def _consume_mprime_output(self, out):
        """
        daemon function to consume and operate on subprocess output
        """
        def log_uncaught_output(output):
            print('uncaught output: {}'.format(output))

        for binary_line in iter(out.readline, b''):
            line = binary_line.decode().rstrip('\n')

            if line.startswith('[Main thread'):
                if 'Starting workers.' in line:
                    self.status = Statuses.running
                    continue
                if 'Stopping all worker threads.' in line:
                    self.status = Statuses.stopping
                    for worker in self.workers:
                        worker.status = Statuses.stopping
                    continue

            elif line.startswith('[Worker'):
                worker_number = int(line.split()[1].strip('#'))
                if not self.workers.get(worker_number):
                    worker = Worker(worker_number)
                    self.workers[worker_number] = worker
                    self.handlers['on_worker_add'](worker)
                else:
                    worker = self.workers[worker_number]

                if 'Worker starting' in line:
                    worker.status = Statuses.starting
                    self.handlers['on_worker_start'](worker)
                    continue
                elif 'Beginning a continuous self-test on your computer.' in line:
                    continue
                elif 'Please read stress.txt.  Hit ^C to end this test.' in line:
                    continue
                elif 'Lucas-Lehmer iterations' in line:  # beginning of a task
                    worker.add_test(line.strip('.').split('Test')[-1].split())
                    continue
                elif 'Self-test' in line and 'passed' in line:
                    test_length = line.split('Self-test')[-1].split()[0]
                    if worker.tests[-1].length == test_length:
                        worker.tests[-1].status = Statuses.passed
                        self.handlers['on_worker_pass'](worker)
                        continue
                elif 'Worker stopped' in line:
                    worker.status = Statuses.stopped
                    continue
                # handle failed tests and update objects as messages come in
                elif 'FATAL ERROR:' in line:
                    worker.tests[-1].status = Statuses.failed
                    continue
                elif 'Hardware failure detected,' in line:
                    worker.tests[-1].status_reason = 'hardware failure'
                    worker.status_reason = 'hardware failure'
                    self.handlers['on_worker_fail'](worker)
                    continue
                elif 'Torture Test completed' in line:
                    worker.summary = line.split('Torture Test')[-1].strip()
                    self.handlers['on_test_complete'](worker)
                    continue

            log_uncaught_output(line)

        out.close()
        self.stop_time = time.time()

    def _write_prime_txt(self):
        with open('prime.txt', 'w+') as f:
            f.write(self.dict_to_ini(self.config) + '\n')

    @staticmethod
    def dict_to_ini(in_dict: dict) -> str:
        return '\n'.join('{}={}'.format(k, v) for k, v in in_dict.items())


if __name__ == '__main__':
    mprime = MPrime()
    mprime.handlers['on_worker_start'] = lambda w: print(mprime.statusline())
    mprime.handlers['on_test_complete'] = lambda w: print(w.summary)
    mprime.launch_mprime()
    while mprime.running:
        time.sleep(500)
    mprime.stop()
