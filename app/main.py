from __future__ import annotations

import os
import random
import threading
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_CACHE_DIR = BASE_DIR / "model-cache"

MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
IMAGE_COUNT = int(os.getenv("IMAGE_COUNT", "10"))
STEPS = int(os.getenv("INFERENCE_STEPS", "24"))
GUIDANCE = float(os.getenv("GUIDANCE_SCALE", "7.0"))
WIDTH = int(os.getenv("WIDTH", "512"))
HEIGHT = int(os.getenv("HEIGHT", "512"))

NEGATIVE_PROMPT = (
    "text, letters, watermark, logo, signature, frame, border, people, portrait, "
    "low quality, blurry, malformed architecture, duplicate objects, oversaturated"
)

PLACES = [
    "an empty metaphysical piazza",
    "a deserted theater stage",
    "a silent arcade",
    "an impossible coastal palace",
    "a forgotten railway platform",
    "a red stone courtyard",
    "an abandoned museum hall",
    "an endless architectural corridor",
    "a lonely desert pavilion",
    "a vast room open to the sea",
    "a ruined observatory",
    "a silent town square",
    "an abandoned hotel lobby",
    "a flooded museum gallery",
    "a moonlit cloister",
    "an underground cathedral",
    "a forgotten glass greenhouse",
    "an empty opera foyer",
    "an endless marble gallery",
    "an impossible rooftop garden",
    "a silent astronomical observatory",
    "a deserted cinema auditorium",
    "a windowless chapel",
    "a flooded library reading room",
    "an empty airport terminal",
    "a cavernous underground station",
    "a ruined botanical conservatory",
    "a deserted seaside promenade",
    "a monumental stair hall",
    "an abandoned indoor swimming pool",
    "a hidden monastery courtyard",
    "an ancient archive chamber",
    "a silent sculpture garden",
    "a sealed ballroom",
    "a dreamlike government corridor",
    "an empty department store atrium",
    "a forgotten winter garden",
    "a lonely coastal fortress",
    "a museum basement maze",
    "a deserted university hall",
    "a sunken palace courtyard",
    "an empty grand hotel dining room",
    "a silent factory assembly hall",
    "an abandoned exhibition pavilion",
    "a circular room beneath the earth",
    "a vast hall of identical doors",
    "a narrow alley between impossible buildings",
    "a dry fountain plaza under a pale sky",
    "a monumental terrace above a black sea",
    "an empty station platform inside a palace",
    "a quiet red corridor with no visible end",
    "a marble courtyard suspended above clouds",
    "a forgotten seaside funicular station",
    "an abandoned palace greenhouse at dusk",
    "a silent tunnel lined with classical columns",
    "a vast waiting room with no exits",
    "an underground museum with open ceilings",
    "a deserted amphitheater facing the ocean",
    "a lonely atrium filled with still water",
    "an impossible bridge connecting empty towers",
]
OBJECTS = [
    "floating doors",
    "a faceless marble statue",
    "an empty chair",
    "a giant candle",
    "a suspended staircase",
    "sealed windows",
    "a stone horse",
    "a black sun",
    "a red moon",
    "a broken classical column",
    "a white mannequin",
    "an oversized glove",
    "a solitary clock without hands",
    "a doorway leading into the sky",
    "a floating picture frame",
    "a mirror reflecting a different room",
    "an elevator standing outdoors",
    "an empty glass display case",
    "a black obelisk casting two shadows",
    "a white grand piano with no keys",
    "a bronze bell suspended in midair",
    "a giant key resting on stone",
    "an empty birdcage",
    "a cracked telescope",
    "a marble sphere",
    "a golden cube",
    "a ladder disappearing into fog",
    "a red velvet curtain hanging outdoors",
    "a stone mask on an empty pedestal",
    "an antique globe with blank oceans",
    "a row of identical black umbrellas",
    "a transparent chair",
    "a train carriage with no tracks",
    "a doorframe with no wall",
    "a staircase folded into a circle",
    "a giant chess piece",
    "a silent fountain flowing upward",
    "an open book with blank pages",
    "a white bed in the center of a plaza",
    "a marble hand emerging from the floor",
    "a locked cabinet with glass doors",
    "a chandelier hanging close to the ground",
    "a telephone without a cord",
    "a solitary red suitcase",
    "a bronze horse head",
    "an empty theater seat",
    "a window suspended above the floor",
    "a doorway filled with ocean water",
    "a staircase ending at a ceiling",
    "a giant stone eye",
    "a black cube half buried in sand",
    "a pale balloon tied to a column",
    "a rowboat resting indoors",
    "a mirror facing a blank wall",
    "a narrow bridge leading to nowhere",
    "a fountain filled with black water",
    "a suspended railway signal",
    "a marble bust turned toward the wall",
    "a red chair beneath a vast arch",
    "a white flag hanging without wind",
    "an empty aquarium",
    "a stone table set for nobody",
    "a giant seashell on a pedestal",
    "a small house inside a large room",
    "a glowing doorway beneath the floor",
    "a staircase descending into still water",
    "a row of clocks showing different times",
    "a long dining table with one plate",
    "a black curtain concealing part of the sky",
    "a bronze cage containing only light",
    "a marble arch standing alone",
    "a door casting a shadow in the wrong direction",
    "a white horse statue covered with cloth",
    "a red sphere hovering above tiles",
    "a cracked museum display pedestal",
    "a giant window opening onto darkness",
    "a stone boat floating above the floor",
    "a silent carousel horse",
    "an unlit streetlamp indoors",
    "a row of sealed suitcases",
]
ARCHITECTURE = [
    "long arcades and severe geometric architecture",
    "bent red walls shaped like folded fabric",
    "impossibly deep perspective",
    "monumental arches and narrow towers",
    "smooth buildings with no visible entrances",
    "theatrical walls and sculptural corridors",
    "ancient masonry mixed with organic forms",
    "precise plazas distorted by dream logic",
    "repeating archways that shrink into darkness",
    "colossal blank facades with tiny doors",
    "classical columns arranged in impossible rhythms",
    "low ceilings opening into vast exterior courtyards",
    "windowless towers connected by narrow bridges",
    "marble staircases crossing without meeting",
    "curved museum walls enclosing an empty horizon",
    "symmetrical halls interrupted by one crooked doorway",
    "severe modernist planes beside ruined classical arches",
    "terraces layered like an abandoned stage set",
    "oversized doorways and unnaturally small windows",
    "corridors folding back into themselves",
    "monumental walls pierced by isolated circular openings",
    "a maze of colonnades beneath an open sky",
    "flat geometric buildings casting theatrical shadows",
    "a solitary tower beyond endless arcades",
    "smooth plaster architecture warped by perspective",
    "fragmented palace walls floating above the ground",
    "deep porticoes surrounding a silent central void",
    "angular red structures rising from pale stone",
    "classical facades partially submerged in still water",
    "impossible interior balconies facing one another",
]
LIGHT = [
    "extremely long late-afternoon shadows",
    "dim crimson twilight",
    "cold moonlight through mist",
    "pale morning light with unnatural shadows",
    "a single warm theatrical spotlight",
    "flat silent daylight beneath a green sky",
    "soft violet dusk",
    "low amber light",
    "hard noon light with shadows pointing in different directions",
    "faint blue light rising from the floor",
    "overcast silver daylight",
    "a distant red glow behind the architecture",
    "soft museum light with no visible source",
    "deep twilight broken by one illuminated doorway",
    "dusty golden light through high windows",
    "cold fluorescent light in an otherwise classical room",
    "moonlight reflected from still black water",
    "a pale sunrise beneath a dark sky",
]
PALETTES = [
    "muted mineral colors",
    "deep crimson, dusty pink and slate blue",
    "ochre, rust and pale turquoise",
    "dark red and foggy lavender",
    "earthy brown, faded blue and ivory",
    "black, brick red and ash gray",
    "pale green sky with warm terracotta",
    "desaturated violet and burnt orange",
    "stone gray, faded rose and deep teal",
    "bone white, olive green and oxidized copper",
    "smoky blue, dark umber and muted gold",
    "chalk white, burgundy and charcoal",
    "dusty cyan, pale peach and black",
    "weathered cream, rust red and storm blue",
    "sickly green, sandstone and dark maroon",
    "cold gray, faded yellow and blood red",
]
MOODS = [
    "motionless and ominously quiet",
    "dreamlike and theatrical",
    "mysterious but calm",
    "symbolic and unsettling",
    "lonely, solemn and timeless",
    "quietly absurd",
    "poetic and uncanny",
    "like a memory of a place that never existed",
    "serene but subtly threatening",
    "ceremonial and abandoned",
    "nostalgic for an impossible past",
    "silent as if time has stopped",
    "beautiful, distant and emotionally unreadable",
    "familiar yet spatially wrong",
    "calm with an approaching sense of disaster",
    "like an empty stage after everyone disappeared",
]
STYLE = [
    "metaphysical surrealist oil painting",
    "European surrealist oil painting",
    "symbolic theatrical painting",
    "dreamlike fine-art oil painting",
    "precise architectural surrealism",
    "museum-quality figurative surrealism",
    "early twentieth-century metaphysical painting",
    "quiet liminal surrealism rendered in oil",
    "classical perspective transformed by dream logic",
    "melancholic architectural fantasy painting",
    "theatrical symbolist oil painting",
    "uncanny modernist fine-art painting",
]
TEXTURE = [
    "rich canvas texture",
    "softly blended oil paint",
    "subtle aged varnish",
    "matte painted surface",
    "fine visible brushwork",
    "smooth classical paint handling",
    "slightly cracked antique paint surface",
    "thin translucent oil glazes",
    "dry mineral pigment texture",
    "soft atmospheric brushwork",
]


state_lock = threading.Lock()
generation_lock = threading.Lock()
pipeline_lock = threading.Lock()
pipeline: StableDiffusionPipeline | None = None

state: dict[str, Any] = {
    "running": False,
    "model_loading": False,
    "progress": [0.0] * IMAGE_COUNT,
    "status": ["waiting"] * IMAGE_COUNT,
    "images": [],
    "error": None,
    "generation_id": 0,
}

app = FastAPI(title="Dream Museum", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def random_prompt() -> str:
    selected_objects = random.sample(OBJECTS, k=random.choice([1, 2]))
    return ", ".join([
        random.choice(PLACES), random.choice(ARCHITECTURE), " and ".join(selected_objects),
        random.choice(LIGHT), random.choice(PALETTES), random.choice(MOODS),
        random.choice(STYLE), random.choice(TEXTURE),
        "carefully composed, high detail, no people, no text"
    ])


def get_pipeline() -> StableDiffusionPipeline:
    global pipeline
    with pipeline_lock:
        if pipeline is not None:
            return pipeline
        with state_lock:
            state["model_loading"] = True
        try:
            loaded = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                cache_dir=str(MODEL_CACHE_DIR),
                safety_checker=None,
                requires_safety_checker=False,
            )
            loaded.scheduler = DPMSolverMultistepScheduler.from_config(loaded.scheduler.config)
            loaded = loaded.to("cpu")
            loaded.enable_attention_slicing()
            loaded.enable_vae_slicing()
            loaded.set_progress_bar_config(disable=True)
            pipeline = loaded
            return loaded
        finally:
            with state_lock:
                state["model_loading"] = False


def reset_output() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.glob("*.webp"):
        path.unlink(missing_ok=True)


def generate_exhibition(generation_id: int) -> None:
    try:
        reset_output()
        pipe = get_pipeline()
        for index in range(IMAGE_COUNT):
            with state_lock:
                if state["generation_id"] != generation_id:
                    return
                state["status"][index] = "generating"
                state["progress"][index] = 1.0

            prompt = random_prompt()
            seed = random.SystemRandom().randint(0, 2**32 - 1)
            generator = torch.Generator(device="cpu").manual_seed(seed)

            def callback_on_step_end(_pipe, step_index: int, _timestep, callback_kwargs):
                percent = min(98.0, ((step_index + 1) / STEPS) * 98.0)
                with state_lock:
                    if state["generation_id"] == generation_id:
                        state["progress"][index] = round(percent, 1)
                return callback_kwargs

            result = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                width=WIDTH,
                height=HEIGHT,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=generator,
                callback_on_step_end=callback_on_step_end,
            )
            filename = f"dream_{index + 1:02d}.webp"
            temp_path = OUTPUT_DIR / f".{filename}.tmp"
            final_path = OUTPUT_DIR / filename
            result.images[0].save(temp_path, format="WEBP", quality=92, method=6)
            temp_path.replace(final_path)

            with state_lock:
                state["progress"][index] = 100.0
                state["status"][index] = "complete"
                state["images"] = [f"/images/{p.name}?v={generation_id}" for p in sorted(OUTPUT_DIR.glob("*.webp"))]
    except Exception as exc:
        with state_lock:
            state["error"] = str(exc)
            for i, item_status in enumerate(state["status"]):
                if item_status == "generating":
                    state["status"][i] = "failed"
    finally:
        with state_lock:
            state["running"] = False
        generation_lock.release()




def _icon_response(filename: str, media_type: str | None = None) -> FileResponse:
    path = STATIC_DIR / "icons" / filename
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/site.webmanifest", include_in_schema=False)
@app.get("/manifest.webmanifest", include_in_schema=False)
def webmanifest() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "site.webmanifest",
        media_type="application/manifest+json",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico() -> FileResponse:
    return _icon_response("favicon.ico", "image/x-icon")


@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg() -> FileResponse:
    return _icon_response("favicon.svg", "image/svg+xml")


@app.get("/favicon-16x16.png", include_in_schema=False)
def favicon_16() -> FileResponse:
    return _icon_response("favicon-16x16.png", "image/png")


@app.get("/favicon-32x32.png", include_in_schema=False)
def favicon_32() -> FileResponse:
    return _icon_response("favicon-32x32.png", "image/png")


@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def apple_touch_icon() -> FileResponse:
    return _icon_response("apple-touch-icon.png", "image/png")


@app.get("/icon-96.png", include_in_schema=False)
def icon_96() -> FileResponse:
    return _icon_response("icon-96.png", "image/png")


@app.get("/icon-128.png", include_in_schema=False)
def icon_128() -> FileResponse:
    return _icon_response("icon-128.png", "image/png")


@app.get("/icon-256.png", include_in_schema=False)
def icon_256() -> FileResponse:
    return _icon_response("icon-256.png", "image/png")


@app.get("/icon-384.png", include_in_schema=False)
def icon_384() -> FileResponse:
    return _icon_response("icon-384.png", "image/png")


@app.get("/android-chrome-192x192.png", include_in_schema=False)
def android_192() -> FileResponse:
    return _icon_response("icon-192.png", "image/png")


@app.get("/android-chrome-512x512.png", include_in_schema=False)
def android_512() -> FileResponse:
    return _icon_response("icon-512.png", "image/png")


@app.get("/maskable-192.png", include_in_schema=False)
def maskable_192() -> FileResponse:
    return _icon_response("maskable-192.png", "image/png")


@app.get("/maskable-512.png", include_in_schema=False)
def maskable_512() -> FileResponse:
    return _icon_response("maskable-512.png", "image/png")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    with state_lock:
        return dict(state)


@app.post("/api/generate")
def start_generation() -> dict[str, Any]:
    if not generation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An exhibition is already being generated.")
    with state_lock:
        state["generation_id"] += 1
        generation_id = state["generation_id"]
        state["running"] = True
        state["model_loading"] = pipeline is None
        state["progress"] = [0.0] * IMAGE_COUNT
        state["status"] = ["waiting"] * IMAGE_COUNT
        state["images"] = []
        state["error"] = None
    threading.Thread(target=generate_exhibition, args=(generation_id,), daemon=True).start()
    return {"started": True, "generation_id": generation_id}


@app.get("/images/{filename}")
def image(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    path = OUTPUT_DIR / safe_name
    if not path.exists() or path.suffix.lower() != ".webp":
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/webp", headers={"Cache-Control": "no-store"})
