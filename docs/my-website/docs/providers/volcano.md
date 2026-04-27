# Volcano Engine (Volcengine)
https://www.volcengine.com/docs/82379/1263482

:::tip

**We support ALL Volcengine models including Chat and Embeddings, just set `model=volcengine/<any-model-on-volcengine>` as a prefix when sending litellm requests**

:::

## API Key
```python
# env variable
os.environ['VOLCENGINE_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['VOLCENGINE_API_KEY'] = ""
response = completion(
    model="volcengine/<OUR_ENDPOINT_ID>",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ],
    temperature=0.2,        # optional
    top_p=0.9,              # optional
    frequency_penalty=0.1,  # optional
    presence_penalty=0.1,   # optional
    max_tokens=10,          # optional
    stop=["\n\n"],          # optional
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['VOLCENGINE_API_KEY'] = ""
response = completion(
    model="volcengine/<OUR_ENDPOINT_ID>",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ],
    stream=True,
    temperature=0.2,        # optional
    top_p=0.9,              # optional
    frequency_penalty=0.1,  # optional
    presence_penalty=0.1,   # optional
    max_tokens=10,          # optional
    stop=["\n\n"],          # optional
)

for chunk in response:
    print(chunk)
```

## Sample Usage - Embedding
```python
from litellm import embedding
import os

os.environ['VOLCENGINE_API_KEY'] = ""
response = embedding(
    model="volcengine/doubao-embedding-text-240715",
    input=["hello world", "good morning"]
)
print(response)
```

### Supported Embedding Models
- `doubao-embedding-large` (2048 dimensions)
- `doubao-embedding-large-text-250515` (2048 dimensions)
- `doubao-embedding-large-text-240915` (4096 dimensions)
- `doubao-embedding` (2560 dimensions) 
- `doubao-embedding-text-240715` (2560 dimensions)

### Embedding Parameters
```python
from litellm import embedding

response = embedding(
    model="volcengine/doubao-embedding-text-240715",
    input=["sample text"],
    encoding_format="float",  # optional: "float" (default), "base64"
    user="user-123",          # optional: user identifier for tracking
)
```

## Sample Usage - Video Generation

LiteLLM supports Volcengine Ark Seedance video generation through the `volcengine` provider.
Ark video generation is task-based: create a video task, poll the task status, then download the generated content.

```python
from litellm import video_content, video_generation, video_status
import os
import time

os.environ["VOLCENGINE_API_KEY"] = ""

video = video_generation(
    model="volcengine/doubao-seedance-2-0-260128",
    prompt="A cinematic shot of a tram moving through a neon city street",
    seconds="5",
    size="1280x720",
    user="user-123",
)

while video.status not in ("completed", "failed", "cancelled"):
    time.sleep(10)
    video = video_status(video_id=video.id)

if video.status == "completed":
    video_bytes = video_content(video_id=video.id)
    with open("output.mp4", "wb") as f:
        f.write(video_bytes)
```

LiteLLM maps the common OpenAI-style video parameters to Ark fields:

| LiteLLM parameter | Ark field |
| --- | --- |
| `prompt` | `content` text item |
| `input_reference` | `content` image item |
| `seconds` | `duration` |
| `size` | `resolution` and `ratio` |
| `user` | `safety_identifier` |

### Ark-native video parameters

For Ark-specific workflows, pass native Ark request fields through `extra_body`.
This is the recommended way to use Seedance features such as first/last frame generation,
reference images, reference videos, reference audio, draft tasks, tools, service tiers, and last-frame output.

```python
from litellm import video_generation
import os

os.environ["VOLCENGINE_API_KEY"] = ""

video = video_generation(
    model="volcengine/doubao-seedance-2-0-260128",
    prompt="",
    extra_body={
        "content": [
            {
                "type": "text",
                "text": "Make this scene cinematic with smooth camera motion",
            },
            {
                "type": "image_url",
                "role": "first_frame",
                "image_url": {"url": "https://example.com/first-frame.png"},
            },
            {
                "type": "image_url",
                "role": "last_frame",
                "image_url": {"url": "https://example.com/last-frame.png"},
            },
        ],
        "resolution": "720p",
        "ratio": "16:9",
        "duration": 5,
        "seed": 11,
        "return_last_frame": True,
        "generate_audio": True,
        "service_tier": "default",
        "watermark": True,
    },
)
```

To download the last frame image, create the task with `return_last_frame=True`, wait for completion, then request the `last_frame` variant:

```python
from litellm import video_content

last_frame_bytes = video_content(video_id=video.id, variant="last_frame")
with open("last_frame.png", "wb") as f:
    f.write(last_frame_bytes)
```

### Supported video operations

- `video_generation()` creates an Ark video generation task.
- `video_status()` retrieves task status and metadata.
- `video_content()` downloads the generated video after completion.
- `video_content(..., variant="last_frame")` downloads the last frame image when requested at creation time.
- `video_list()` lists recent Ark video generation tasks.
- `video_delete()` cancels queued tasks or deletes completed/failed/expired task records.

Ark expresses video editing, extension, remix, and multimodal reference workflows through the create-task API and native `content` blocks.
LiteLLM does not expose separate OpenAI-style `video_edit`, `video_extension`, or `video_remix` endpoints for Volcengine.
Use `video_generation(..., extra_body={"content": [...]})` for those Ark-native workflows.

## Supported Models - 💥 ALL Volcengine Models Supported!
We support ALL `volcengine` models for chat completions, embeddings, and Ark video generation:
- **Chat Models**: Set `volcengine/<OUR_ENDPOINT_ID>` as a prefix when sending completion requests
- **Embedding Models**: Use the specific model names listed above (e.g., `volcengine/doubao-embedding-text-240715`)
- **Video Models**: Use Seedance model IDs or endpoint IDs (e.g., `volcengine/doubao-seedance-2-0-260128`, `volcengine/doubao-seedance-2-0-fast-260128`, or `volcengine/<YOUR_SEEDANCE_ENDPOINT_ID>`)

## Sample Usage - LiteLLM Proxy

### Config.yaml setting

```yaml
model_list:
  # Chat model
  - model_name: volcengine-model
    litellm_params:
      model: volcengine/<OUR_ENDPOINT_ID>
      api_key: os.environ/VOLCENGINE_API_KEY
  # Embedding model
  - model_name: volcengine-embedding
    litellm_params:
      model: volcengine/doubao-embedding-text-240715
      api_key: os.environ/VOLCENGINE_API_KEY
  # Video generation model
  - model_name: volcengine-video
    litellm_params:
      model: volcengine/doubao-seedance-2-0-260128
      api_key: os.environ/VOLCENGINE_API_KEY
```

### Send Request

#### Chat Completion
```shell
curl --location 'http://localhost:4000/chat/completions' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "volcengine-model",
    "messages": [
        {
        "role": "user",
        "content": "here is my api key. openai_api_key=sk-1234"
        }
    ]
}'
```

#### Embedding
```shell
curl --location 'http://localhost:4000/embeddings' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "volcengine-embedding",
    "input": ["hello world", "good morning"]
}'
```

#### Video Generation
```shell
curl --location 'http://localhost:4000/v1/videos' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "volcengine-video",
    "prompt": "A cinematic shot of a tram moving through a neon city street",
    "seconds": "5",
    "size": "1280x720"
}'
```

Check task status:

```shell
curl --location 'http://localhost:4000/v1/videos/{video_id}' \
    --header 'Authorization: Bearer sk-1234'
```

Download the generated video after completion:

```shell
curl --location 'http://localhost:4000/v1/videos/{video_id}/content' \
    --header 'Authorization: Bearer sk-1234' \
    --output output.mp4
```

Cancel or delete a task:

```shell
curl --request DELETE 'http://localhost:4000/v1/videos/{video_id}' \
    --header 'Authorization: Bearer sk-1234'
```
