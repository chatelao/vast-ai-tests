from aiohttp import web
import json
import asyncio

async def get_models(request):
    return web.json_response({
        "object": "list",
        "data": [
            {
                "id": "tiny-model",
                "object": "model",
                "created": 1677610602,
                "owned_by": "organization-owner"
            }
        ]
    })

async def chat_completions(request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    stream = data.get("stream", False)

    if not stream:
        return web.json_response({
            "choices": [{"message": {"content": "This is a mock response."}}]
        })

    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        },
    )
    await response.prepare(request)

    tokens = ["This", " is", " a", " mock", " streaming", " response."]
    for token in tokens:
        chunk = {
            "choices": [{"delta": {"content": token}}]
        }
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode('utf-8'))
        await asyncio.sleep(0.05) # Simulate some ITL

    await response.write(b"data: [DONE]\n\n")
    return response

app = web.Application()
app.router.add_get('/v1/models', get_models)
app.router.add_post('/v1/chat/completions', chat_completions)

if __name__ == '__main__':
    print("Starting mock server on port 8000...")
    web.run_app(app, port=8000)
