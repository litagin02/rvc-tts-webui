from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

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


@app.post("/tts")
async def gura(options: RvcOptions | None = None):
    try:
        (
            model_name,
            speed,
            tts_text,
            tts_voice,
            f0_key_up,
            f0_method,
            index_rate,
            protect0,
        ) = options
        info_text, edge_tts_output, tts_output = tts(
            model_name,
            speed,
            tts_text,
            tts_voice,
            f0_key_up,
            f0_method,
            index_rate,
            protect0,
        )
        return FileResponse(tts_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
