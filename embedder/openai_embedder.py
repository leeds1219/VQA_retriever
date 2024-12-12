from openai import OpenAI
import openai
from dotenv import load_dotenv
import asyncio
import uuid
import json
from io import BytesIO
import json
import uuid
import time
import json
import aiohttp
import numpy as np

from embedder.base_embedder import BaseEmbedder

class RetrieveException(Exception):
    pass
    
def parse_output_file(text):
    count = 0
    prev = 0
    arr = []
    for i in range(len(text)):
        if text[i] == '{':
            count += 1
        elif text[i] == '}':
            count -= 1
            if count == 0:
                arr.append(text[prev:i+1])
                prev = i+1
    try:
        result = list(map(json.loads, arr))
    except Exception as error:
        for _ in arr:
            print(_)
        raise error
    return result

def batch_request(client, end_point, batch, custom_batch_id):
    
    # Attempt batch request creation with retry mechanism
    for _ in range(25):
        try:
            # Upload batch file
            file_object = client.files.create(
                file=BytesIO(batch.encode()),
                purpose="batch"
            )
            
            # Create batch processing request
            batch_object = client.batches.create(
                input_file_id=file_object.id,
                endpoint=end_point,
                completion_window="24h",
                metadata={
                    "custom_batch_id": custom_batch_id,
                }
            )
            
            print(f"################# Requested: {batch_object.id} {custom_batch_id} ######################")
            break
        except Exception as error:
            print(f"Error creating batch: {error}")
            time.sleep(1.5 ** _)

    return batch_object
        


async def wait_retrieve(client, batch_id: str, description: str='', delay=300, max_hr=24):
    retry = max_hr * 60 * 60 * 1.5 // delay
    
    headers = {
        "Authorization": f"Bearer {client.api_key}"
    }
    
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        # pbar = tqdm(range(retry), total=retry,)
        # pbar.set_description(f"Waiting for batch api..." + (" ({description})" if description else ""))
        # for _ in pbar:
        start_time = time.time()
        duration = max_hr * 60 * 60 * 1.5
        while True:
            time_passed = time.time() - start_time
            try:
                async with session.get(f"{client.base_url}/batches/{batch_id}") as resp:
                    request = await resp.json()
                    if resp.status != 200:
                        continue
                    if request['status'] in ["failed", "error", "cancelling", "cancelled", "expired"]:
                        raise RetrieveException("RetrieveException: " + str(request))
                    elif request['status'] == "completed" and request["output_file_id"] is not None:
                        print(f"{batch_id} completed")
                        async with session.get(f"{client.base_url}/files/{request['output_file_id']}/content") as file_resp:
                            output_file = await file_resp.text()
                            return parse_output_file(output_file)
                    elif request['status'] not in ['in_progress', 'finalizing']:
                        pass
                if time_passed > duration:
                    break
                seconds_passed = time_passed // 60
                await asyncio.sleep(delay)
                print(f"Waiting for batch api..." + (" ({description})" if description else "") + f"({seconds_passed} minutes passed)")
            except RetrieveException as e:
                raise e
            except:
                delay = min(delay * 2, 60*60)
                continue

    raise TimeoutError

def list_requests(client: OpenAI):
    current_time = int(time.time())
    time_threshold = current_time - 24 * 60 * 60
    print(time_threshold)

    for request in client.batches.list():
        if request.status in ['in_progress', 'finalizing', 'finalizing', 'cancelling']:
            print(request.id, request.status, request.request_counts, (current_time - request.created_at) // 60)
            # print(request)
        # if request.created_at > time_threshold:
        #     break
                
def cancel_all_requests(client: OpenAI):
    time_threshold = int(time.time()) - 24 * 60 * 60
    
    for request in client.batches.list():
        if request.status in ["in_progress", "finalizing"]:
            custom_batch_id = request.metadata["custom_batch_id"] if "custom_batch_id" in request.metadata else ""
            print(f"Cancelling {request.id} {custom_batch_id}...", end="\t")
            print(request)
            try:
                client.batches.cancel(request.id)
                print(f"success")
            except Exception as error:
                print("failed")
                print(error)
        # if request.created_at > time_threshold:
        #     break
        

class OpenAIEmbedder(BaseEmbedder):

    def __init__(self, model_name='text-embedding-3-small', use_batch_api=False, processor_config=None):

        load_dotenv()
        self.client = OpenAI()
        self.batch_url = "/v1/embeddings"

        if processor_config is None:
            processor_config = {}
            
        # OpenAI max allowed batch size is 50_000
        if 'batch_size' not in processor_config:
            processor_config['batch_size'] = 50_000 
            self.batch_size = 1_000
        else:
            self.batch_size = processor_config['batch_size'] = min(processor_config['batch_size'], 50_000 if use_batch_api else 1_000)
            
        super().__init__(model_name, use_batch_api, processor_config)


    async def _batch_api(self, texts):
        
        batch = []
        for i, text in enumerate(texts):
            batch.append({
                "custom_id": str(i),
                "method": "POST",
                "url": self.batch_url,
                "body": {"input": [text], "model": self.model_name}
            })
        
        # Encode batch as JSONL
        batch = "".join([json.dumps(line) + "\n" for line in batch])

        custom_batch_id = str(uuid.uuid4())
        
        for _ in range(20):
            try:
                print(f"{len(batch)=}")
                batch_object = batch_request(self.client, self.batch_url, batch, custom_batch_id)
                outputs = await wait_retrieve(self.client, batch_object.id)
                break
            except RetrieveException as e:
                print(e)
                time.sleep(60 * (2 ** _))
                continue

        # Parse results
        results = [None for _ in range(len(texts))]
        for output in outputs:
            custom_id = int(output["custom_id"])
            results[custom_id] = output["response"]["body"]["data"][0]["embedding"]
        return results


    async def __call__(self, texts):
        
        if self.use_batch_api:
            tasks = []
            for text in texts:
                task = asyncio.create_task(self.processor.fetch(text))
                tasks.append(task)
            results = await asyncio.gather(*tasks)
        else:
            results = []
            batch_size = 1000
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                for _ in range(25):
                    try:
                        print(f"########################## Sending OpenAI API embedding request ({i+batch_size}/{len(texts)}) ##########################")
                        response = self.client.embeddings.create(
                            input=batch_texts,
                            model=self.model_name
                        )

                        batch_results = [None for _ in range(len(texts))]
                        for embedding in response.data:
                            batch_results[embedding.index] = embedding.embedding
                        results += batch_results
                        break
                    except Exception as e:
                        if any(isinstance(e, cls) for cls in (
                            openai.APIError,
                            openai.RateLimitError,
                            openai.APIConnectionError,
                            openai.Timeout,
                            openai.InternalServerError,
                        )):
                            print(e)
                            time.sleep(10 * (1.5 ** _))
                            continue
                        raise e
                
        return np.array(results)
    
if __name__ == "__main__":
    client = OpenAI()
    list_requests(client)
    # cancel_all_requests(client)
    