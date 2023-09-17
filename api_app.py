from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import argparse
import uvicorn


from tts import tts


app = FastAPI()


class RvcOptions(BaseModel):
    model_name: str
    speed: int
    tts_text: str
    tts_voice: str
    f0_key_up: int
    f0_method: str
    index_rate: int
    protect0: float


def yield_audio(audio_arr):
    yield from audio_arr


@app.post("/tts")
def convert_text_to_rvc_speech(options: RvcOptions | None = None):
    try:
        info_text, edge_tts_output, tts_output = tts(
            options.model_name,
            options.speed,
            options.tts_text,
            options.tts_voice,
            options.f0_key_up,
            options.f0_method,
            options.index_rate,
            options.protect0,
        )
        print(f"info_text: ${info_text}")
        print(f"edge_tts_output: ${edge_tts_output}")
        print(f"tts_output: ${tts_output}")
        return StreamingResponse(yield_audio(tts_output[1]), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "main":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    uvicorn.run(app, port=8001)
