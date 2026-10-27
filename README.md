# ComfyUI Direct for AstrBot

Klein image generation and editing, with background jobs and **plugin-owned image delivery**.
The LLM requests an image; Python submits it, monitors completion and sends the attachment
back through the original AstrBot message event. No local paths, placeholder tokens,
member shell permission or second LLM send call are needed.

**Status: v0.3.0, persistent image reuse.** SDK interfaces
reviewed against AstrBot **4.29.3**; workflow nodes checked against ComfyUI **0.42.0**.
Automated tests use mock services/events. Text generation and QQ attachment delivery
have passed a manual deployment smoke test. Cross-turn image reuse and editing in this
release still need a live smoke test.

## What works

- Text-to-image with distilled FLUX.2 Klein 4B: four steps, Euler, batch one.
- Reference-conditioned generation and image edits, with up to three ordered images.
- A persistent image library for incoming attachments, resolved quotes, generated outputs
  and public URL references. All use opaque `img_...` handles scoped to user and conversation.
- Recent image IDs are added to each LLM request. Listing supports pagination for older images.
- Imports from search/other tools can be saved before generating, then reused without
  downloading the external image again. Input bytes are uploaded to ComfyUI per job.
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
   | `quiet_image_replies` | Default true: accepted image tools end the agent turn; background delivery continues |
   | `image_retention_days` | Reusable image lifetime; default 30 days, range 1–365 |

5. Enable/reload the plugin after saving the endpoint. The default endpoint is deliberately
   blank. The plugin loads without it so settings remain accessible; image tools report
   that configuration is needed until a valid endpoint has been saved.
6. Start a new AstrBot conversation to remove old tool instructions/context. Leave member
   shell execution disabled. No changes to the Portainer stack or extra ports are needed.
7. Try “Generate a red fox in a warm bookstore.” Verify you receive an **image attachment**.
8. Without reattaching the generated image, ask “Make the background a snowy forest,
   keep the subject.” The edit must report a nonempty `input_image_ids` in its tool result.
   Also test an uploaded image, then refer to it by its library ID in a later turn.
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

- `comfy_generate_image(prompt, reference_urls=[], use_attached_images=false, image_ids=[])`
- `comfy_edit_image(instruction, reference_urls=[], use_attached_images=true, image_ids=[])`
- `comfy_list_images(limit=20, offset=0)`
- `comfy_import_image(reference_url, label="")`
- `comfy_image_status(job_id="", retry_delivery=false)`

Generated outputs receive stable image IDs as soon as they are downloaded, independently
of the LLM turn. Job status returns output IDs; the next request's context includes recent
images, their provenance and labels. Labels are request/source descriptions, **not verified
visual captions**. They are marked as untrusted metadata in the injected context.

Explicit `image_ids` take precedence over auto-attached inputs; URLs are appended in order.
With no explicit inputs, editing first uses current/quoted attachments. If none exist,
it uses the requesting user's most recent generated image in this conversation. A latest
job with multiple outputs requires an explicit selection; unknown/foreign/expired IDs fail
without falling back to a different image. For a less specific or ambiguous request, the
agent should list images and clarify the intended input.

Example: generate a fox → receive `img_...` in the library → edit that ID with “change the
background to snow.” This is image-conditioned editing, not a new text-only approximation.
The tool result includes the actual `input_image_ids` and `reference_count`. After an edit
fails validation, text-only generation is blocked for the remainder of that same incoming
request. The agent must obtain a valid image or ask for clarification; a fresh user request
can still intentionally request a new image.

The LLM request hook imports up to three current/resolved-quoted images, even before a
user requests generation, and attaches only this user's scoped metadata to the prompt.
Another group member cannot reuse those handles merely by knowing an ID. Explicit
cross-user sharing is not implemented; the recipient must reattach an image themselves.

An empty status ID lists your five most recent jobs in the current conversation.
With `quiet_image_replies=true` (default), successful generation/edit tools return no
text. AstrBot ends the agent turn instead of asking the model for another status
announcement. Validation errors still return normally; background failures still notify
the requesting conversation. Library IDs and status remain available in later turns.
Disable this option for workflows that need additional LLM/tool steps after submission,
such as iterative inspection. In diagnostic mode (`quiet_image_replies=false`) the tool
returns its job ID and state as before.

This prevents post-submission LLM narration; it cannot recall text the model already
sent before the tool call. For fewer chat bubbles, use AstrBot's full tool-schema mode,
disable tool-status display, and consider non-streaming responses with intermediate
message buffering. Buffering merges messages; it does not itself remove verbose text.
An optional concise character persona is in [personas/erika-zh.md](personas/erika-zh.md).

The plugin sends completion independently of the LLM turn, including if another message arrives. Do not configure an agent rule that
calls `send_message_to_user` for these results or waits using the old plugin's tools.

No manual persona/system-prompt changes are required. The plugin supplies tool schemas
and scoped image context automatically. Input resolution, authorization, missing-input
errors, duplicate suppression, expiry and delivery are enforced in Python. The LLM still
interprets the user's intent, selects tools and writes conversational text; its wording is
not a transport receipt. The following optional persona guidance is only for tone:

> Use comfy_generate_image or comfy_edit_image for image requests. The plugin delivers
> the image itself. A queued job is not a delivered image. Do not print filesystem paths,
> ask for shell/admin access, or claim an image was sent before delivery. Use shared
> search tools for research; use real reference images when visual fidelity is requested.

Quoted images work only when the platform adapter supplies an `Image` in `Reply.chain`.
An unresolved QQ quote ID cannot be turned into an image by this plugin; ask the user to
reattach the image. Account-avatar retrieval is not implemented. No OneBot-specific API
is assumed. References from an older message can be reattached/quoted or provided as a URL;
older images can instead be selected by ID from `comfy_list_images`. There is no global
“last image” shared between users.

## Recovery and failure behavior

Jobs and generated PNG files live in AstrBot's plugin data directory, persisted with its
`data` volume. These files contain user content and should be included in your retention
and backup policy. Reusable library copies expire after `image_retention_days`; expired
library files/records are deleted on startup and LLM requests. Access does not renew expiry.
Original job-delivery files and job records retain the existing manual retention policy;
changing library retention does not delete those files or ComfyUI's server-side outputs.
The full workflow graph is removed after download, but a short request description is
retained for image selection. ComfyUI separately retains its inputs, outputs and history
according to its own policy. Backups can retain files after local expiry.

Existing databases migrate automatically. Still-present, non-expired output files from
older jobs are indexed on startup. Persistent IDs survive restarts; a missing/expired image
must be reattached/imported rather than guessed or replaced.

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

Tests cover cross-turn generated-image reuse, persistence and legacy database migration,
image expiry, attachment capture without generation, URL import/reuse, scoped model context,
pagination, ambiguous latest outputs, conversation/user isolation, duplicate submissions, attachment extraction,
missing edit inputs, backend failures, queue bounds, restart states, delivery retries,
image decoding, URL/DNS restrictions and the HTTP upload/prompt/history/view contract.
Adapter tests use doubles, not a running AstrBot installation. No automated test sends
QQ messages. The live installation checklist above is required before production use.
