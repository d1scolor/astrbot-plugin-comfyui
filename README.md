# ComfyUI Direct for AstrBot

Klein image generation and editing, with background jobs and **plugin-owned image delivery**.
The LLM requests an image; Python submits it, monitors completion and sends the attachment
back through the original AstrBot message event. No local paths, placeholder tokens,
member shell permission or second LLM send call are needed.

**Status: first implementation, ready for a controlled deployment test.** SDK interfaces
reviewed against AstrBot **4.29.3**; workflow nodes checked against ComfyUI **0.42.0**.
Automated tests use mock services/events. This replacement has not yet been tested live
on QQ. Enable it in a test conversation before replacing production use.

## What works

- Text-to-image with distilled FLUX.2 Klein 4B: four steps, Euler, batch one.
- Reference-conditioned generation and image edits, with up to three ordered images.
- Images from the current message, quoted message chains when supplied by the adapter,
  or public image URLs returned by other tools.
- Immediate completion checks, then one-second polling; no fixed 30/60-second wait.
- Persistent jobs, one GPU job at a time, bounded queue, two active jobs per user/conversation.
- Ownership checks include both sender and originating conversation. A later group message
  cannot redirect an existing job. Identical tool requests within one message are deduplicated.
- Restart recovery resumes queued/running jobs without resubmitting acknowledged prompts.
- No search engine or extra management web server. Use AstrBot's shared search/vision tools
  to research a character, then pass a description and/or actual public reference-image URLs.

Search results alone do not mean a reference image was used. The tool reports
`reference_count`. Selecting the right image is the agent's responsibility.

## Install and replace the old plugin

1. Back up AstrBot's persistent data using your usual deployment backup procedure.
2. Disable the previous ComfyUI plugin in AstrBot. Keep its files/configuration for rollback.
   Do not leave both plugins' generation tools enabled in the same test conversation.
3. In AstrBot's plugin management, install from:
   `https://github.com/d1scolor/astrbot-plugin-comfyui`
4. Configure **ComfyUI Direct** in the plugin settings:

   | Setting | Value |
   | --- | --- |
   | `comfy_url` | ComfyUI HTTP(S) base URL reachable **from the AstrBot container** |
   | `api_token` | Optional bearer token, only if your reverse proxy requires one |
   | `model` | `flux-2-klein-4b.safetensors` |
   | `encoder` | `qwen_3_4b.safetensors` |
   | `vae` | `flux2-vae.safetensors` |
   | `width`, `height` | Default 1024; multiples of 16, between 256 and 1536 |
   | `job_timeout` | Default 600 seconds, including queue time |
   | `queue_limit` | Default 8 unfinished jobs |

5. Enable/reload the plugin after saving the endpoint. The default endpoint is deliberately
   blank. The plugin loads without it so settings remain accessible; image tools report
   that configuration is needed until a valid endpoint has been saved.
6. Start a new AstrBot conversation to remove old tool instructions/context. Leave member
   shell execution disabled. No changes to the Portainer stack or extra ports are needed.
7. Try “Generate a red fox in a warm bookstore.” Verify you receive an **image attachment**.
8. Attach a picture and ask “Make the background a snowy forest, keep the subject.”
   Try a group request and send a follow-up while generation runs. Verify the image returns
   to the requesting group and requests in another conversation stay separate.

Use the distilled model named above; base models, other architectures, video and arbitrary
workflow imports are not supported in this version. Model licenses still apply.
The old plugin's custom Simple String node and named workflows are not used.

The bundled [text-to-image workflow](workflows/klein-text-to-image.api.json) is the starting
graph. Editing adds core `LoadImage`, `ImageScaleToTotalPixels`, `VAEEncode` and
`ReferenceLatent` nodes for each input. It follows the conditioning mechanism in
[ComfyUI's official distilled Klein edit template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_flux2_klein_image_edit_4b_distilled.json).
Output dimensions follow plugin settings; reference images are resized to roughly one
megapixel. This is generative editing, not a pixel-preserving editor. Multi-reference
quality and edited outputs still need live visual validation on your deployment.

## Tools and conversation behavior

- `comfy_generate_image(prompt, reference_urls=[], use_attached_images=false)`
- `comfy_edit_image(instruction, reference_urls=[], use_attached_images=true)`
- `comfy_image_status(job_id="", retry_delivery=false)`

An empty status ID lists your five most recent jobs in the current conversation.
The tool returns a job ID promptly. The plugin sends completion independently of the
LLM turn, including if another message arrives. Do not configure an agent rule that
calls `send_message_to_user` for these results or waits using the old plugin's tools.

Suggested persona addition, if your model needs it:

> Use comfy_generate_image or comfy_edit_image for image requests. The plugin delivers
> the image itself. A queued job is not a delivered image. Do not print filesystem paths,
> ask for shell/admin access, or claim an image was sent before delivery. Use shared
> search tools for research; use real reference images when visual fidelity is requested.

Quoted images work only when the platform adapter supplies an `Image` in `Reply.chain`.
An unresolved QQ quote ID cannot be turned into an image by this plugin; ask the user to
reattach the image. Account-avatar retrieval is not implemented. No OneBot-specific API
is assumed. References from an older message must be reattached/quoted or provided as a URL;
there is no implicit “last image” shared between users.

## Recovery and failure behavior

Jobs and generated PNG files live in AstrBot's plugin data directory, persisted with its
`data` volume. These files contain user content and should be included in your retention
and backup policy. There is no automatic deletion in this first version. Workflow prompts
are removed from the database after output is downloaded successfully. ComfyUI separately
retains its inputs, outputs and history according to its own policy.

| State | Meaning / action |
| --- | --- |
| `queued`, `running` | Automatically resumed after restart; ComfyUI must retain the prompt history. |
| `generated` | Image saved locally. If its event was lost in a restart, ask to check this job's status to deliver it using the new request. |
| `delivered` | AstrBot's send method returned successfully. This is not a user read receipt. |
| `submission_unknown` | Server may have accepted the request before connection/crash failure. Not automatically resubmitted; inspect ComfyUI before requesting a new generation. |
| `delivery_unknown` | Sending raised or AstrBot restarted during delivery. Ask to retry delivery explicitly; it might duplicate a partially accepted send. |
| `failed` | Execution/history/output retrieval failed. Check ComfyUI and request a new job after fixing the cause. |
| `timed_out` | Monitoring deadline exceeded. ComfyUI may still finish; inspect it before generating again. |

Checking status does not resend an already `delivered` job. Explicit delivery retry uses
`retry_delivery=true`; it never regenerates. Requests cannot access another user's job,
even in a shared group session. Completed jobs recovered without an event are delivered
only when their owner checks them from the same conversation.

QQ platform permissions, quotas, passive-reply windows and upstream adapter behavior still
apply. The plugin uses `event.send`, not filesystem-restricted agent tools. Some upstream
adapter failure paths can return without raising; inspect AstrBot logs if a job says
`delivered` but no image arrives. Exact-once delivery cannot be guaranteed across a crash
or an uncertain transport response. No automatic retry masks that uncertainty.

A monitoring timeout or disabling/reloading this plugin **does not interrupt ComfyUI's
shared GPU execution**. Other clients' jobs must not be interrupted. Per-job cancellation
and cross-process/multiple-AstrBot-instance coordination are not implemented.

## Security and development

Only the administrator-configured ComfyUI endpoint can access a private network.
Reference URLs must be public HTTP(S), port 80/443, without credentials. DNS addresses
are validated at connection time and redirects are revalidated. Downloads are limited
to 20 MiB; decoded images are limited to 25 megapixels and re-encoded as RGB PNG.
Input metadata and transparency are removed. Tool arguments cannot select local files.
Local image components are accepted only from AstrBot's platform event.

The ComfyUI endpoint is trusted infrastructure. Keep it private or authenticated;
this plugin does not secure the server itself. Do not publish real endpoints, tokens,
job databases, generated images or diagnostic logs in this repository.

```sh
uv sync --group dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Tests cover conversation/user isolation, duplicate submissions, attachment extraction,
missing edit inputs, backend failures, queue bounds, restart states, delivery retries,
image decoding, URL/DNS restrictions and the HTTP upload/prompt/history/view contract.
Adapter tests use doubles, not a running AstrBot installation. No automated test sends
QQ messages. The live installation checklist above is required before production use.
