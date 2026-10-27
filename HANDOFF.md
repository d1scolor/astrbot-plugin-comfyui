# Project handoff

## Decision and scope

Build a new, focused AstrBot plugin. Reuse AstrBot's QQ
adapter, personas, LLM tools and attachment APIs, and ComfyUI's existing HTTP
and WebSocket APIs. Do not build a chatbot framework, model server or search
engine. The first implementation is present; see README.md for current capabilities and deployment validation.

Initial operations:

- Generate an image from a text prompt, optionally with reference images.
- Edit supplied images using a separate, explicitly configured workflow.
- Resolve current-message and quoted images where the platform supports it.
- Track completion and send images directly to the originating conversation.

General web search belongs to AstrBot's shared tool layer. A separate image
search/vision capability may be required to find and select character
references. Search results or a character description alone do not establish
that a generation used a reference image. Avatar retrieval depends on the
official QQ adapter/API's actual capabilities; do not assume OneBot features.

## Deployment assumptions

- AstrBot and ComfyUI may run on separate Docker hosts.
- Target integration is the official QQ WebSocket adapter, not OneBot/NapCat.
- ComfyUI runs as a non-root service with persistent model/output storage.
- Target versions are AstrBot 4.29.3 and ComfyUI 0.42.0; text generation and QQ delivery passed a manual smoke test. v0.2.0 image reuse still needs live validation.
- Service endpoints and credentials must be supplied through local configuration,
  never committed. Do not assume access to a contributor's deployment.
- Keep GPU selection and host provisioning outside this plugin.

No credentials, host addresses, hardware identifiers, account IDs, raw logs,
or diagnostic exports should be committed here.

## Working model and workflow

Models already downloaded using host hf CLI installed with uv:

- diffusion_models/flux-2-klein-4b.safetensors (BFL BF16 distilled model).
- text_encoders/qwen_3_4b.safetensors.
- vae/flux2-vae.safetensors.

The core-only API workflow is in workflows/klein-text-to-image.api.json.
Node 4.inputs.text is the positive prompt; node 5.inputs.noise_seed is the
seed. Width/height must be updated together in nodes 8 and 9. Node 12 saves
the result. It uses Euler, Flux2Scheduler, four steps, 1024x1024, batch one.
This is text-to-image only. The implementation builds editing/reference nodes in engine/workflow.py; visual output
validation on the target deployment is still required.

A sample workflow has successfully generated images in a development
installation. Performance depends on hardware, precision, caching and workload;
no portable latency guarantee or stock/unlocked comparison is claimed.

## Why replace the current plugin

Current plugin: https://github.com/cjxzdzh/astrbot_plugin_comfyui , version
1.0.1 in logs. Its behavior was inspected in source and observed in logs.

1. It returns a local path and relies on the LLM to send the image, or to
   output a special placeholder. A completed group job was sent as a plain
   path instead of an attachment after a follow-up arrived during the wait.
2. A hardcoded developer-machine download path violated member file
   permissions. The user patched it to /AstrBot/data/temp/comfyui and
   restarted. Private-chat image sending then succeeded.
3. query_wait deliberately sleeps before checking completion. Source default
   and minimum are 30 seconds; the model requested 60 seconds in logs.
   A patch capping this to two seconds was provided, but the exported logs
   still show actual 60-second sleeps; application of that patch is unverified.
4. It requires custom Simple String nodes and filename conventions. A minimal
   custom node was installed for compatibility. Our new plugin should use
   explicit API input mappings and not require that custom node.
5. Tasks and follow-ups must be owned by immutable originating conversation
   and request identifiers, not a model-supplied user ID or global fallback.

Do not solve delivery errors by granting members shell execution or admin.
Do not mistake the model's explanation of a failure for the actual tool error.

## Desired design

- Small LLM tool surface; plugin owns submit, wait/track and deliver.
- No arbitrary filesystem paths returned to the LLM for copying/sending.
- Use AstrBot-managed storage/media APIs and validated image handles.
- Treat user-provided URLs and files as untrusted; bound download sizes and
  types, restrict filesystem access and prevent arbitrary internal URL fetches.
- Distinguish generated, delivered, failed and queued states accurately.
- No fixed 30/60-second sleeps. Use completion events or bounded polling.
- Maintain per-job/per-conversation ownership and avoid duplicate delivery.
- Support recoverable jobs across restart, timeouts and delivery failures.
- Preserve image quality and provide clear errors for unsupported workflows.
- No extra management website initially. Use plugin configuration for workflow
  paths/mappings and trusted service endpoints.
- Search and character research compose with generation as separate tools.

## Validation before replacement

Test text-to-image, image editing, current/quoted image inputs, private/group
delivery, simultaneous users/groups, follow-ups during generation, ComfyUI
errors, QQ send failure, timeout, cancellation policy, duplicate notifications
and restart recovery. Verify with member permissions and no shell tools.

Use local/mock tests first, then targeted tests on an explicitly authorized test deployment.
Do not send unsolicited QQ messages. Enable only one generation integration
for the test conversation to prevent competing tools/hooks.

## Alternatives investigated

- nagato9star/astrbot_plugin_kuyo_comfyflow: persistent jobs, generation/edit
  routing, character/resource lookup; small adoption footprint, not audited
  or tested on the target official QQ + Klein setup.
- lxmsqm/astrbot-omnipotent-comfyui: broad functionality, direct delivery
  described; many QQ features refer to OneBot/NapCat, compatibility unverified.
- WalkerZJH/astrbot_plugin_easy_comfyui: small command-based integration;
  inspected main file did not expose LLM tool decorators.
- Open WebUI's documented ComfyUI generation/editing integration is a design
  reference, not a drop-in QQ transport.

Do not call these alternatives proven based on stars, README claims or
marketplace inclusion. The existing plugin was initially recommended too
quickly in the originating chat.

## Documentation starting points

- https://docs.astrbot.app/en/dev/star/plugin-new.html
- https://docs.astrbot.app/en/dev/star/guides/send-message.html
- https://docs.astrbot.app/en/dev/star/guides/ai.html
- https://github.com/AstrBotDevs/AstrBot
- https://docs.comfy.org/development/comfyui-server/comms_routes
- https://docs.comfy.org/tutorials/flux/flux-2-klein

Verify documentation routes and SDK APIs against deployed 4.29.3; these
links are starting points, not guarantees of a stable interface.

## v0.2.0 image library

Persistent image IDs now cover generated outputs, attachments/resolved quotes and public
URL imports. Jobs migrate automatically and index older available outputs on recovery.
The LLM request hook adds scoped recent IDs; generation/edit tools accept explicit IDs,
and editing defaults to the user's latest single generated output when no inputs exist.
List/import tools support external tool composition without local paths. Retention applies
to library copies; original job-delivery files still have manual retention. See README.

## v0.4.0 additions

Deterministic caption/crop/resize operations live in engine/postprocess.py. Captions use
bundled OFL Noto CJK and Pillow; ComfyUI core nodes composite with inverted LoadImage
mask. Generation/edit accepts postprocess; comfy_process_image skips diffusion.
comfy_send_images delivers selected public URLs or owned image IDs without generation.
Optional deliver=false retains intermediates; status never auto-sends these. Existing
SQLite databases migrate kind/auto_deliver columns. New behavior needs AstrBot smoke
testing; a synthetic caption passed live ComfyUI validation without sending QQ messages.
