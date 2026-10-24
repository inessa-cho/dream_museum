# Dream Museum

A CPU-based Stable Diffusion web app that creates temporary exhibitions of ten surrealist paintings.

## UI

- **Generate tab**: one Generate button and ten independent progress bars.
- **Exhibition tab**: only the current generated works, displayed in simple frames.
- No prompts, seeds, metadata, favorite system, login, or database are shown or stored.
- Starting a new generation deletes the previous ten images. Images remain only until the next generation or manual deletion.

## Run

```bash
cp .env.example .env
docker compose up -d --build
```

Open:

```text
http://SERVER_IP:8012
```

View logs:

```bash
docker compose logs -f dream-museum
```

Stop:

```bash
docker compose down
```

## Notes

The first run downloads the SD1.5 model into `model-cache/`. On a CPU-only 8–12 GB machine, each image may take a long time. The ten works are generated sequentially so memory usage stays reasonable. Progress is updated at every diffusion step for the currently active work.


## Implementation notes

The FastAPI app generates ten SD1.5 images sequentially on CPU and exposes progress and current-exhibition views. A new generation replaces previous exhibition images. Pinned requirements and Docker definitions are retained; no model download, image generation or new build was run.

## Models and assets

Model weights and generated gallery images are not included. The configured model is downloaded separately; its license and usage restrictions apply when running the generator. The current code disables the model safety checker. A fresh model download and generation run have not been verified.

See [PRESERVATION.md](PRESERVATION.md) for the maintenance record, excluded files and verification scope.
