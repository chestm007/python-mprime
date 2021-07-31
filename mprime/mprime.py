import os
import time
from enum import Enum
from subprocess import Popen, PIPE
from threading import Thread
from typing import List, Union


class CONFIGURATIONS:
    SMALLEST_FFTS = dict(
        StressTester=1,
        UsePrimenet=0,
        MinTortureFFT=4,  # in K
        MaxTortureFFT=13,  # in K
        TortureMem=0,  # in MiB. value is per thread if 8 or less, torture test does FFTs in-place
        TortureTime=6,  # in minutes
        TortureThreads=None,  # if None detect automatically
    )

    SMALL_FFTS = dict(
        StressTester=1,
        UsePrimenet=0,
        MinTortureFFT=22,  # in K
        MaxTortureFFT=85,  # in K
        TortureMem=0,  # in MiB. value is per thread if 8 or less, torture test does FFTs in-place
        TortureTime=6,  # in minutes
        TortureThreads=None,  # if None detect automatically
    )

    LARGE_FFTS = dict(
        StressTester=1,
        UsePrimenet=0,
        MinTortureFFT=146,  # in K
        MaxTortureFFT=8192,  # in K
        TortureMem=29461,  # in MiB. value is per thread if 8 or less, torture test does FFTs in-place
        TortureTime=6,  # in minutes
        TortureThreads=None,  # if None detect automatically
    )

    BLEND = dict(
        StressTester=1,
        UsePrimenet=0,
        MinTortureFFT=4,  # in K
        MaxTortureFFT=8192,  # in K
        TortureMem=29461,  # in MiB. value is per thread if 8 or less, torture test does FFTs in-place
        TortureTime=6,  # in minutes
        TortureThreads=None,  # if None detect automatically
    )


Statuses = Enum("STATUSES", "stopped starting running passed failed stopping")


def mk_esc(esc_chars):
    return lambda s: "".join(["" if c in esc_chars else c for c in s])


class DataSize:
    unit_string = "TGMKB"

    def __init__(self, size: Union[int, str]):
        self.value = self._parse_string(size)
        self._raw_value = size

    def _parse_string(self, in_string):
        esc = mk_esc(self.unit_string)
        base_int = int(esc(in_string))
        for i, unit in enumerate(self.unit_string):
            if unit in in_string:
                base_int *= 1024 ** (len(self.unit_string) - (i + 1))
        return base_int

    def __eq__(self, other):
        if isinstance(other, DataSize):
            return other.value == self.value
        return other == self.value

    def __str__(self):
        return self._raw_value


class Test:
    def __init__(
        self,
        test: int,
        num_iterations: int,
        thing: str,
        using: str,
        length: str,
        passes: list = None,
        clm=None,
    ):
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
        self.tests: List[Test] = []
        self.status: Union[Statuses, None] = None
        self.status_reason = None
        self.summary = None

    @staticmethod
    def __chunked_line_parser(chunked_line: List[str]):
        test = chunked_line.pop(0).strip(",")
        num_iterations = chunked_line.pop(0)
        test_name = chunked_line.pop(0)
        _filler_iterations = chunked_line.pop(0)
        _filler_of = chunked_line.pop(0)
        thing = chunked_line.pop(0)

        using_a = using_b = type_num = type_name = length = None

        if "using" in chunked_line[0]:
            _filler_using = chunked_line.pop(0)
            using_a = chunked_line.pop(0)
            using_b = chunked_line.pop(0)
        else:
            raise RuntimeError("Unexpected property in Test case")

        if "type-" in chunked_line[0]:
            type_num = chunked_line.pop(0)

        type_name = chunked_line.pop(0)

        if "length" in chunked_line[0]:
            _filler_length = chunked_line.pop(0)
            length = chunked_line.pop(0).strip(",")

        chunked_line = [i.strip(",") for i in chunked_line]

        return (
            test,
            num_iterations,
            thing,
            using_a,
            using_b,
            type_num,
            type_name,
            length,
            chunked_line,
        )

    def add_test(self, chunked_line: List[str]):
        """
        parse a line of output representing a test start and add it to the workers list of tests
        """
        (
            test,
            num_iterations,
            thing,
            using_a,
            using_b,
            _,
            _,
            length,
            args,
        ) = self.__chunked_line_parser(chunked_line)
        if args:
            clm = args.pop(-1)
        test_obj_params = [
            test,
            num_iterations,
            thing,
            "{} {}".format(using_a, using_b),
            DataSize(length),
        ] + ([[args], clm] if args else [])
        self.tests.append(Test(*test_obj_params))
        self.status = Statuses.running


class MPrime:
    executable = "mprime"

    handlers = dict(
        on_worker_pass=lambda w: w,
        on_worker_fail=lambda w: w,
        on_worker_start=lambda w: w,
        on_worker_add=lambda w: w,
        on_test_complete=lambda w: w,
        on_test_start=lambda w: w,
        on_uncaught_output=lambda o: print("uncaught output: {}".format(o)),
    )

    def __init__(
        self, prime_config: dict = None, working_directory: str = "/tmp/mprime"
    ):
        self.mprime = None  # type: Popen
        self.output_thread = None
        self.start_time = None
        self.stop_time = None
        self.status = Statuses.stopped
        self.workers = {}

        self.config = CONFIGURATIONS.BLEND
        if prime_config is not None:
            self.config.update(prime_config)

        self.working_directory = working_directory
        if not os.path.isdir(self.working_directory):
            os.mkdir(self.working_directory)
        os.chdir(self.working_directory)

    def launch_mprime(self):
        self._write_prime_txt()
        self.mprime = Popen(
            [self.executable, "-t"], stdout=PIPE, stderr=PIPE, stdin=PIPE
        )
        self.start_time = time.time()

        self.output_thread = Thread(
            target=self._consume_mprime_output, args=(self.mprime.stdout,)
        )
        self.output_thread.daemon = True  # thread dies with the program
        self.output_thread.start()

    def statusline(self):
        output_format = "Status: {status}, Workers: ({workers_summary}), Time elapsed: {time_elapsed}"
        workers_summary = "#:{num_workers}, R:{failed_workers}".format(
            num_workers=len(self.workers),
            failed_workers=len(
                [w for w in self.workers.values() if w.status == Statuses.running]
            ),
        )
        return output_format.format(
            status=self.status,
            workers_summary=workers_summary,
            time_elapsed=self.time_elapsed,
        )

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
        if self.mprime:
            self.mprime.kill()
        while self.running:
            time.sleep(0.1)
        self.status = Statuses.stopped

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
        for binary_line in iter(out.readline, b""):
            line = binary_line.decode().rstrip("\n")
            if not line:
                continue

            if line.startswith("[Main thread"):
                if "Starting workers" in line:
                    self.status = Statuses.running
                    continue
                elif "Stopping all worker threads" in line:
                    self.status = Statuses.stopping
                    for worker in filter(
                        lambda w: w.status != Statuses.failed, self.workers.values()
                    ):
                        worker.status = Statuses.stopping
                    continue
                elif "Execution halted." in line:
                    continue

            elif line.startswith("[Worker"):
                worker_number = int(line.split()[1].strip("#"))
                if not self.workers.get(worker_number):
                    worker = Worker(worker_number)
                    self.workers[worker_number] = worker
                    self.handlers["on_worker_add"](worker)
                else:
                    worker = self.workers[worker_number]

                if "Worker starting" in line:
                    worker.status = Statuses.starting
                    self.handlers["on_worker_start"](worker)
                    continue
                elif "Beginning a continuous self-test" in line:
                    continue
                elif "Please read stress.txt.  Hit ^C to end this test." in line:
                    continue
                elif "Setting affinity" in line:
                    continue
                elif "Lucas-Lehmer iterations" in line:  # beginning of a task
                    worker.add_test(line.strip(".").split("Test")[-1].split())
                    self.handlers["on_test_start"](worker)
                    continue
                elif "Self-test" in line and "passed" in line:
                    test_length = DataSize(line.split("Self-test")[-1].split()[0])
                    if worker.tests[-1].length == test_length:
                        worker.tests[-1].status = Statuses.passed
                        self.handlers["on_worker_pass"](worker)
                        continue
                    else:
                        print(worker.tests[-1].length, test_length)
                elif "Worker stopped" in line:
                    worker.status = Statuses.stopped
                    continue
                # handle failed tests and update objects as messages come in
                elif "FATAL ERROR:" in line:
                    worker.tests[-1].status = Statuses.failed
                    continue
                elif "Hardware failure detected," in line:
                    worker.tests[-1].status_reason = "hardware failure"
                    worker.status_reason = "hardware failure"
                    self.handlers["on_worker_fail"](worker)
                    continue
                elif "Torture Test completed" in line:
                    worker.summary = line.split("Torture Test")[-1].strip()
                    self.handlers["on_test_complete"](worker)
                    continue

            self.handlers["on_uncaught_output"](line)

        out.close()
        self.stop_time = time.time()

    def _write_prime_txt(self):
        with open("prime.txt", "w+") as f:
            f.write(self.dict_to_ini(self.config) + "\n")

    @staticmethod
    def dict_to_ini(in_dict: dict) -> str:
        return "\n".join(
            "{}={}".format(k, v) for k, v in in_dict.items() if v is not None
        )


def main():
    mprime = MPrime()
    mprime.handlers["on_test_start"] = lambda w: print(mprime.statusline())
    mprime.handlers["on_test_complete"] = lambda w: print(
        "worker {}: {}".format(w.number, w.summary)
    )
    mprime.launch_mprime()
    try:
        while mprime.running:
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    mprime.stop()


if __name__ == "__main__":
    main()
