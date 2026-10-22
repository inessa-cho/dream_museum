# Dream Museum

CPU-only Stable Diffusion 1.5 app that creates a temporary exhibition of ten randomly composed metaphysical/surrealist paintings.

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


## Preservation implementation notes

The FastAPI app generates ten SD1.5 images sequentially on CPU and exposes progress and current-exhibition views. A new generation replaces previous exhibition images. Pinned requirements and Docker definitions are retained; no model download, image generation or new build was run.

## Maintenance status

Development and maintenance are discontinued. This is a preservation snapshot, not a new release or feature-development plan. Running services and scheduled jobs are not automatically shut down by this status.

This repository is intended for Private preservation. Public release requires a separate review and approval. See PRESERVATION.md for excluded material and verification limits.
