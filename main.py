import os
import queue
import time
import logging
import threading
import multiprocessing
from abc import abstractmethod
from collections import deque

FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)

class ComputeCore:
    def __init__(self):
        self._in_data_queue = None
        self._out_data_queue = None
        self._stop_event = multiprocessing.Event()
        self._process = multiprocessing.Process(target=self._compute_worker, daemon=True)

    def connect_input_queue(self, data_queue):
        self._in_data_queue = data_queue

    def connect_output_queue(self, data_queue):
        self._out_data_queue = data_queue

    def start(self):
        self._process.start()

    def stop(self):
        self._stop_event.set()
        self._process.join()

    def wait(self):
        self._process.join()

    @abstractmethod
    def _compute_worker(self):
        pass

class ConvCore(ComputeCore):
    def __init__(self, kernel, bias=0):
        self._kernel = kernel
        self._bias = bias
        super().__init__()

    def _compute_worker(self):
        logger.debug(f'PID: {os.getpid()} worker init')
        while not self._stop_event.is_set():
            result = 0
            try:
                for i in range(len(self._kernel)):
                    val = self._in_data_queue.get(timeout=1)
                    logger.debug(f'PID: {os.getpid()} popped: {val}')
                    if val is None:
                        self._out_data_queue.put(None)
                        logger.debug(f'PID: {os.getpid()} worker finished')
                        return
                    result += self._kernel[i] * val

                result += self._bias
                self._out_data_queue.put(result)
                logger.debug(f'PID: {os.getpid()} computed result: {result}')

            except queue.Empty:
                logger.debug(f'PID: {os.getpid()} in data queue empty')
                continue

        logger.debug(f'PID: {os.getpid()} worker exit')

class ScalarCore(ComputeCore):
    def __init__(self, scale, offset):
        super().__init__()
        self._scale = scale
        self._offset = offset

    def _compute_worker(self):
        logger.debug(f'PID: {os.getpid()} worker init')
        while not self._stop_event.is_set():
            try:
                val = self._in_data_queue.get(timeout=1)
            except queue.Empty:
                logger.debug(f'PID: {os.getpid()} in data queue empty')
                continue
            logger.debug(f'PID: {os.getpid()} popped: {val}')
            if val is None:
                self._out_data_queue.put(None)
                logger.debug(f'PID: {os.getpid()} worker finished')
                return
            result = self._scale * val + self._offset
            time.sleep(0.01)
            self._out_data_queue.put(result)
            logger.debug(f'PID: {os.getpid()} computed result: {result}')

        logger.debug(f'PID: {os.getpid()} worker exit')

class ComputeDevice:
    def __init__(self, cores, max_in_queue_size=10, max_out_queue_size=10):
        self._data_queues = []
        n = len(cores)
        for i in range(n + 1):
            if i == 0:
                maxsize = max_in_queue_size
            elif i == n:
                maxsize = max_out_queue_size
            else:
                maxsize = 0
            self._data_queues.append(multiprocessing.Queue(maxsize=maxsize))

        self._input_queue = self._data_queues[0]
        self._output_queue = self._data_queues[-1]
        self._cores = cores

        for i, data_queue in enumerate(self._data_queues):
            if i == 0:
                self._cores[i].connect_input_queue(data_queue)
            elif i == len(self._data_queues) - 1:
                self._cores[i - 1].connect_output_queue(data_queue)
            else:
                self._cores[i - 1].connect_output_queue(data_queue)
                self._cores[i].connect_input_queue(data_queue)

    def start(self):
        for core in self._cores:
            core.start()

    def stop(self):
        for core in self._cores:
            core.stop()

    def wait(self):
        for core in self._cores:
            core.wait()

    def stream_input(self, data, timeout=None):
        self._input_queue.put(data, timeout=timeout)

    def stream_output(self, timeout=None):
        return self._output_queue.get(timeout=timeout)

class SyncExecutor:
    def __init__(self, device): 
        self._device = device

    def run(self, feed):
        if not isinstance(feed, (list, tuple)):
            feed = [feed]
        self._device.start()
        out = []
        for val in feed:
            self._device.stream_input(val)
            output = self._device.stream_output()
            if output is None:
                break
            out.append(output)
        return out

class AsyncExecutor:
    def __init__(self, device, input_callback, output_callback): 
        self._device = device
        self._stop_event = threading.Event()
        self._input_cb = input_callback
        self._output_cb = output_callback
        self._outputs = []
        self._input_worker = threading.Thread(target=self._input_worker, daemon=True)
        self._output_worker = threading.Thread(target=self._output_worker, daemon=True)

    def start(self):
        self._device.start()
        self._input_worker.start()
        self._output_worker.start()

    def wait(self):
        self._output_worker.join()
        self._input_worker.join()

    def _input_worker(self): 
        while not self._stop_event.is_set():
            data = self._input_cb()
            self._device.stream_input(data)
            if data is None:
                return

    def _output_worker(self):
        logger.debug('output worker started')
        while not self._stop_event.is_set():
            try:
                out = self._device.stream_output(timeout=1)
            except queue.Empty:
                logger.debug('output worker timed out')
                continue
            logger.debug(out)
            if out is None:
                return
            self._output_cb(out)
        logger.debug('output worker exited')

class AsyncApp:
    def __init__(self, device, data):
        self._data_iter = iter(data)
        self._outputs = []
        self._async_executor = AsyncExecutor(device, self._get_data, self._recv_output)

    def run(self):
        self._async_executor.start()
        self._async_executor.wait()
        return self._outputs

    def _get_data(self):
        try:
            return next(self._data_iter)
        except StopIteration:
            return None

    def _recv_output(self, out):
        self._outputs.append(out)

def run_sync(device, data):
    sync_executor = SyncExecutor(device)
    return sync_executor.run(data)

def run_async(device, data):
    async_app = AsyncApp(device, data)
    return async_app.run()

if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    cores = [ScalarCore(1.1, 0) for i in range(10)]
    device = ComputeDevice(cores)
    data = list(range(100))
    t = time.time()
    output = run_async(device, data)
    print('Execution time: ', round(time.time() - t, 2))
    logger.info(output)

