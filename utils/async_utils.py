import inspect
from typing import Any, Callable, List
import asyncio


class AsyncBatchProcessor:
    def __init__(
        self,
        process_batch: Callable[[List[Any]], List[Any]],
        batch_size: int=10_000,
        batch_timeout: float=1.0
    ):
        """
        Initialize the batch processor.
        
        :param batch_size: Maximum number of requests in a batch.
        :param batch_timeout: Timeout in seconds to process a batch.
        :param process_batch: A function that processes a batch of requests and returns responses.
        """
        if not inspect.iscoroutinefunction(process_batch):
            raise ValueError

        self._batch_size = batch_size
        self._batch_timeout = batch_timeout # since latest add
        self.process_batch = process_batch
        
        self.request_queue = asyncio.Queue()
        self.task = None  # Store the processor task to ensure it's in the correct loop

    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def batch_timeout(self):
        return self._batch_timeout
    
    @batch_size.setter
    def batch_size(self, new_value):
        if not isinstance(new_value, (int, float)):
            raise ValueError
        self._batch_size = new_value
    
    @batch_timeout.setter
    def batch_timeout(self, new_value):
        if not isinstance(new_value, (int, float)):
            raise ValueError
        self._batch_timeout = new_value

    async def fetch(self, request: Any) -> Any:
        """
        Add a request to the queue and get the response asynchronously.
        
        :param request: The request to be processed.
        :return: The response corresponding to the request.
        """
        future = asyncio.get_event_loop().create_future()
        await self.request_queue.put((request, future))
        return await future

    async def _batch_processor(self):
        """
        Process batches of requests in parallel based on the defined criteria.
        """
        processing_tasks = []  # To store batch processing tasks

        while True:
            batch = []
            futures = []
            
            try:
                # Wait for the first request with timeout
                item = await asyncio.wait_for(self.request_queue.get(), timeout=None)
                batch.append(item[0])
                futures.append(item[1])
            except asyncio.TimeoutError:
                continue

            # Set a timeout for the remaining requests
            timeout_start = asyncio.get_event_loop().time()

            # Collect additional requests up to batch size within self.timeout
            while len(batch) < self.batch_size:
                remaining_time = self.batch_timeout - (asyncio.get_event_loop().time() - timeout_start)
                if remaining_time <= 0:
                    break

                try:
                    item = await asyncio.wait_for(self.request_queue.get(), timeout=remaining_time)
                    batch.append(item[0])
                    futures.append(item[1])
                    timeout_start = asyncio.get_event_loop().time()
                except asyncio.TimeoutError:
                    break

            # Start a new task to process the batch
            processing_task = asyncio.create_task(self._process_single_batch(batch, futures))
            processing_tasks.append(processing_task)

            # Clean up completed tasks
            processing_tasks = [task for task in processing_tasks if not task.done()]

    async def _process_single_batch(self, batch, futures):
        """
        Process a single batch and resolve corresponding futures.
        """
        try:
            responses = await self.process_batch(batch)  # Process batch asynchronously
            for future, response in zip(futures, responses):
                future.set_result(response)
        except Exception as e:
            for future in futures:
                future.set_exception(e)


    def start(self):
        """
        Start the batch processor in the current event loop.
        """
        if not self.task:
            self.task = asyncio.create_task(self._batch_processor())

    async def stop(self, cancel=True):
        """
        Stop the batch processor gracefully.
        """
        if self.task:
            if cancel:
                self.task.cancel()  # Cancel the main processor
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None