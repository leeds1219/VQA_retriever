from utils.async_utils import AsyncBatchProcessor

class BaseEmbedder(object):

    def __init__(self, model_name, use_batch_api, processor_config=None):
        self.model_name = model_name
        
        self.processor = AsyncBatchProcessor(
            process_batch=self._batch_api,
            **processor_config if processor_config is not None else {}
        )
        self._use_batch_api = use_batch_api
        if use_batch_api:
            self.processor.start()


    async def _batch_api(self, values):
        raise NotImplementedError

    async def __call__(self, prompts, system_message):
        raise NotImplementedError

    
    @property
    def use_batch_api(self):
        return self._use_batch_api
    
    @use_batch_api.setter
    async def use_batch_api(self, value):
        self._use_batch_api = value
        if value:
            self.processor.start()
        else:
            await self.processor.stop()
        
    @property
    def batch_size(self):
        return self.processer.batch_size
    
    @batch_size.setter
    def batch_size(self, new_value):
        self.processor.batch_size = new_value

    @property
    def batch_timeout(self):
        return self.processor.batch_timeout
    
    @batch_timeout.setter
    def batch_timeout(self, value):
        self.processor.batch_timeout = value