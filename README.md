# H3 Motion Context

Chain MiniMax H3 clips so motion and sound keep going across the cut.

The **MiniMax H3 Looping Sampler** runs that chain inside one execution.
Give it the first stock H3 AV latent, a sampler and sigmas, and the number of
tiles. Optional `tile_latent_0`, `tile_latent_1`, ... inputs let each tile use
its own H3 length; all tile latents must share the first tile's spatial shape.
It samples every tile before loading the VAEs, carries the previous joint
video/audio latent directly into the next tile, trims the pinned head from
both decoded streams, and returns one synchronized clip.

The default conditioning supplies the first anchor of tile 1 and the last
anchor of the final tile. Add `tile_conditioning_N` sockets when a tile needs
its own prompt or FL2VA end frame. Build those inputs with the stock MiniMax H3
conditioning nodes: this preserves both the VAE anchor and the text encoder's
image tokens. A tile-specific first anchor is ignored after tile 1 because the
sampled motion context owns that boundary.

The dynamic sockets are zero-based: `tile_conditioning_0` belongs to tile 1,
`tile_conditioning_1` to tile 2, and so on. The matching
`tile_latent_N`/`tile_conditioning_N` names form a pair for tile N+1. The node
shows one labeled group anchor for each autogrow family; that anchor is not a
second conditioning slot. Tiles without an override use the default. If an
override has a last frame, that frame is the tile's hard FL2VA endpoint and its
generated tail becomes the next tile's motion context.
Ref2VA conditionings and their image/video/audio references are supported, but
Ref2VA references remain whole-clip guidance rather than hard endpoints.

Every tile follows H3's `17k+5` frame grid. With the recommended 22-frame
context, the first tile contributes its full length and each later tile adds
that tile's length minus 22. The output `frame_count` reports the exact
assembled length.

`settling_tail_frames` defaults to 0. Set it to a non-negative multiple of 17
when an intermediate tile has a hard last-frame image that makes H3 settle:
the tile is cut at the earlier motion frame, and that same AV latent prefix is
used as the next tile's context. The setting applies only to anchored
intermediate tiles; unanchored motion and the final tile are left intact. The
audio prefix uses H3's 40 Hz grid (`round(frames * 40 / 24)`), while decoded
audio is accumulated with the same global sample rounding as the video.

The sampler's `decode_mode` defaults to `auto`. After sampling and immediately
before decoding, it uses the current VAE memory estimate to choose the largest
safe spatial tile from 256 through 2048 pixels, then uses that choice for every
tile in the run. `advanced` uses the selected `decode_tile_size` instead. H3's
native temporal decoder remains fixed at its 17-frame chunking; changing that
would alter the VAE's latent-to-frame mapping rather than simply changing its
memory use.

The looping node currently supports batch size 1. It returns the last sampled
joint AV latent as `last_tile_latent`, so another run can continue from the
actual model output without a decode/re-encode round trip.

## Example workflow

Load [`example_workflows/minimax_h3_looping.json`](example_workflows/minimax_h3_looping.json)
after installing this pack and restart ComfyUI if necessary. It demonstrates
three variable-length tiles (124, 141, and 158 frames) with the recommended
22-frame context. The settling-tail control defaults to 0 in the sample;
after wiring hard last-frame images to intermediate tiles, set it to 17 to
demonstrate the earlier motion cut. The H3 example
uses one global text node, three tile text nodes, and three
`StringConcatenate` nodes. Each concatenated prompt feeds its own stock H3
Image-to-Video conditioning node, so every tile has independent `first_frame`
and `last_frame` image sockets. The first reference also feeds
**MiniMax H3 Resolution From Image**, whose selected megapixel target supplies
the same width and height to all three H3 nodes. Each H3 latent is connected
to its matching zero-based `tile_latent_N` socket.

The example assumes the standard H3 filenames used by the official workflow:
the FL2VA diffusion model, Qwen3-VL text encoder, and separate video/audio
VAEs. Change those loader widgets if your files use different names.

Connect the first reference image to both the resolution node and tile 1's
`first_frame`. Connect any later tile references to that tile's own image
sockets. The resolution node preserves the first image's aspect ratio while
rounding all three H3 canvases to the selected megapixel target and 32-pixel
grid.

Generate clip A. Feed its last frames and audio into this node. Generate
clip B. B picks up where A left off: same motion, same speed, same
direction, and the same audio continued rather than a new take that
sounds similar. Repeat as long as you like.

Nothing on disk is edited. The nodes patch ComfyUI at runtime, and the
patches check their own math against the live ComfyUI code every time you
start. If an update breaks an assumption, the nodes refuse to run and say
why. A loud failure beats a bad render you don't notice.

### Global and tile prompts

For image-conditioned tiles, use the pattern in the example: keep the common
facts in one `PrimitiveStringMultiline` node, keep each standalone tile
description in its own text node, and join each pair with
`StringConcatenate`. Each resulting string is encoded by that tile's stock H3
Image-to-Video node. There is one active global prompt, not a second hidden
global prompt in the latent-shape node.

Do not write later prompts as "continue the previous tile" or "the same"; the
model receives no previous tile text. Write every tile as a self-contained H3
prompt. For strict base-mode H3 formatting, start the shared global text with
`integrated_multimodal_description:` and let each tile append its own `[Shot 1]`
description followed by exactly one `overall_soundscape:` and one
`non_diegetic_music:` field. The example workflow uses this form.

For an intermediate tile with a hard end image, make the image and prompt
describe active motion at the boundary: constant subject speed, rotating
wheels or an ongoing gait, persistent spray/cloth/hair motion, and camera
movement maintained through the final frame. Avoid endpoint wording that
suggests completion or rest, including "finishes", "arrives", "reaches",
"settles", "centered", "balanced composition", "slows", or fading music or
engine sound. This reduces anticipatory deceleration but cannot override a
last image that depicts rest; choose an intermediate last keyframe with
mid-motion visual cues. The final tile may intentionally settle.

**MiniMax H3 Multi-Prompt Provider** remains available for text-only tiles. It
prepends one global prompt to a `|`-separated prompt list and returns the
conditioning list directly, but it cannot carry per-tile image tokens.

If the prompt list is shorter than the requested tile count, the provider's
final item is repeated. For hard FL2VA endpoints or Ref2VA references, always
connect stock H3 conditioning nodes to the sampler's zero-based
`tile_conditioning_N` sockets; those explicit conditionings override the
provider for their tile. ComfyUI's current autogrow schema cannot nest a
repeating latent-and-conditioning pair, so the sampler uses the matching names
above instead of adding a wrapper node per tile.

## Why this exists

LTX has clip chaining built in. H3 doesn't, but the parts were already
there. H3 can pin a frame at a time coordinate and re-inject it at every
sampling step. The only thing stopping a whole run of frames was one
check in ComfyUI that rejected any pinned frame other than the first or
last. The math already worked for everything in between. This lifts that
check.

Audio was the harder half, and it's the more useful half, since H3
generates picture and sound together. See "Why the audio needed work"
below if you care how.

## Install

Drop the folder in `ComfyUI/custom_nodes/` and restart. At startup you'll
just see:

```
h3_motion_context: nodes registered. ComfyUI patches install on the first
run of a Motion Context node.
```

The patches don't go in until you actually run one. Having the pack
installed changes nothing about your other H3 workflows. The first time
you chain a clip you'll see:

```
h3_motion_context: interior keyframe anchors enabled
h3_motion_context: keyframe/ref coexistence enabled
```

Anything else and the node refuses to run. The reason is logged.

## Wiring

```
MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo (or the t2v path)
  -> H3 Motion Context      <- previous clip's latent (picture + sound)
  -> guider / sampler
  ...
  decoded IMAGE + AUDIO
  -> H3 Motion Context Trim         <- wire trim_frames across
  -> Create Video / save
```

Wire `trim_frames` into the Trim node. The pinned frames come back at the
start of the new clip and have to come off before you concatenate, picture
and sound together.

### Carrying the previous clip across

The previous clip reaches the node as a latent, but you can't wire the sampler
straight into `context_latent`. ComfyUI will call it a circular
connection, and it's right: the latent you want is from the previous run,
not this one. Two helper nodes move it across runs the same way your
frames and audio already move across, through a file:

```
this run:   SamplerCustomAdvanced -> H3 Motion Context Save Latent
next run:   H3 Motion Context Load Latent -> context_latent
```

Both have a `clip_index` and the numbers mean what they say. On Load, the
clip you're continuing FROM. On Save, the clip this one IS. Making clip 2
from clip 1: Load 1, Save 2. Don't like it? Queue again, change nothing.
The retry reloads clip 1 and overwrites clip 2. Happy with it? Bump both
numbers and move on. Files are named the obvious way,
`clip_00002.safetensors` is clip 2.

Leave context unwired for clip 1.

At `clip_index` 0 the loader grabs the newest file in the folder instead.
Simpler, but a re-roll then loads its own rejected audio, so don't use it
for anything you're going to retry. Auto-saved files get a trailing
underscore (`clip_00002_.safetensors`) because they're numbered by run,
not by clip, and indexed loading skips them on purpose.

You can also point the loader at a specific file, which ignores the index.
Its output is only for `context_latent`. Don't wire it into a decode node.
Stock Save/Load Latent won't work here, it can't handle H3's paired
video/audio latent.

The latent carries both streams, so with it wired you don't need to load
the previous clip's video at all. The pinned frames are sliced straight
out of it rather than decoded to pixels and encoded again, so they're
exactly what the model made instead of a reconstruction. Nothing shifts
the colour or the contrast, so there's no seam to see, and it's faster.

Nothing to set for any of that. Leave `context_frames` unwired. Resolution
has to match between clips, since a latent can't be resized; if it doesn't
the node refuses and names both resolutions rather than quietly dropping
to the lossy path.

There's an older path for graphs with no latent: the previous clip's
frames into `context_frames`, its decoded audio into `context_audio`, and
the H3 audio VAE into `audio_vae`. It works, but it costs a lossy round
trip per link on both streams and it's where the visible seams came from.
Use the latent unless you have a reason not to.

### Reference mode

A Ref2VA graph carries its own reference blocks (image, video, video with
audio, audio) and chaining needs one too, for the pinned sound. They share
one list. Put this node after `MiniMaxH3ReferenceToVideo` and wire it as
above. Your references are kept and the continuation audio is added to
them.

Nothing to configure. Worth mentioning only because older versions
overwrote that list, so turning chaining on quietly threw your references
away.

## Settings

Two, because everything else had exactly one right answer.

**context_length** - frames of the previous clip's picture to carry over.
5, 22, 39 or 56. Those are the lengths that are a whole number of latent
steps, which is why the others aren't offered. 5 is just barely fluid, 22
is nearly seamless. Longer windows pin more motion but they come off the
front of the delivered clip, so 56 spends 2.3 seconds of every render on
frames you throw away. **Use 22.**

**audio_context_length** - frames of tail audio to pin, independent of the
picture window. It ends at the same instant as the pinned video, so this
only controls how far back the sound reaches. 0 follows context_length.
**Use 22** to line it up with a 22-frame picture window. Longer windows
(44, 96) are legal but nobody has rendered one.

Everything else is fixed: the pinned run is encoded in one VAE call, it
sits at the head of the clip where the Trim node removes it, and the
pinned audio goes on this clip's own timeline. The alternatives all
existed only to reproduce their own failures, so they're constants at the
top of `nodes.py` now. Change one there if you ever need to see what they
did.

`match_tail` on the Trim node stays a setting because that node has no
idea what the other one did. Leave it on. H3 rounds its audio grid up, so
every clip carries about 8ms more sound than picture, and that error
stacks at every join.

## Writing prompts for a chain

The settings get motion and sound across the join. What happens in the
next clip is on you, and there are a few traps.

**The model renders contradictions as unions.** Clip N ends on a close-up
of A, clip N+1's prompt opens with "a two-shot of B and C," and the model
doesn't pick. You get all three. The pinned frames aren't a suggestion, so
a prompt describing a different arrangement of people reads as an addition
to them, not a replacement.

**The airlock.** Don't ask for the change and the continuation at the same
moment. Open clip N+1 holding clip N's exact closing framing, no dialogue,
about two seconds, then cut to the new setup. Joins done this way measure
tighter than an ordinary frame-to-frame cut. Joins that skip it measure
like two different rooms spliced together.

**Give the hold something to do.** A held framing with nothing happening
renders as a literal freeze, and two seconds of a motionless actor looks
like the video stalled. Write in a breath, a weight shift, an eyeline
change. The camera holds still, the performer doesn't.
`tests/freeze_detect.py` finds these.

**Budget the pinned head in your timecodes.** In `head` mode the clip
comes out `context_length` frames shorter than it was sampled. At 22 that's
0.92 seconds. Your prompt timings land against the sampled version, which
starts 0.92s earlier, so a beat you wrote for 4.0s shows up at 3.08s in the
file.

## Why the audio needed work

The first version ran pinned audio through H3's reference mechanism, which
is where audio conditioning normally goes. Every join had a small tick,
like the audio briefly sped up and went offbeat. Looking at the waveform
showed nothing wrong. Both sides of every join were smooth on their own.

Cross-correlating each clip's opening against the previous clip's ending
(that's `tests/seam_probe.py`) showed what was actually happening: the new
clip's audio resembled the old one. Same instruments, same groove, never
the same recording. A cover band. The model was reading the reference as
"a separate clip that sounds like this," which is what references are for
and exactly wrong for continuation.

The fix is the same one that already worked for video. The rows the model
sees are identical either way. What differs is their time coordinates, and
the coordinates are what say "separate clip" versus "this clip, earlier."
So the pinned audio still rides the reference machinery, but its
coordinates get rewritten onto the new clip's timeline, ending exactly
where the pinned video ends. Correlation at the joins went from about 0.45
with incoherent timing to 0.95+ with a flat offset, and the tick was gone.
Measured across a chain, the offset doesn't grow from join to join.

## Limitations

**Quality degrades down a chain.** The big one, and it's mostly audio.
Each clip is generated from the previous clip's output, which came from
the one before it. Losses compound like photocopying a photocopy, and in
audio the top end goes first. Timing and tempo stay locked, but after
several clips the sound gets duller and more muffled. Picture holds up
much better.

Two things stack per link: the model's own smoothing, and a round trip
out to pixels and back. `context_latent` removes the second one for both
streams. On picture that's the difference between a visible join and no
join at all. On sound it helps, but the model's own smoothing is still
there. Long chains are worth listening to critically, and restarts land
best at a natural musical transition.

**A small audio offset, now believed fixed.** Chained clips used to come
out about 8ms late. The cause was arithmetic: a frame is 5/3 of an audio
step, so unless the pinned window's length works out to a whole number of
steps it landed between the ones the model was filling. The window is now
snapped to that grid. The offset was constant, well under where lip-sync
errors get noticeable, and it didn't grow down a chain, so this is a
tidy-up rather than a rescue. `tests/seam_probe.py` reports the lag in
milliseconds if you want to check it on your own material.

**H3 emits 32 kHz audio, not 48.** Read the rate off the clip in any script
that remuxes or concatenates. A stream-copy concat can't change rate
partway, so a hardcoded 48000 silently turns the tail of a long episode
into nothing while every duration check still passes.
`tests/level_step.py` prints every clip's rate and flags a mismatch.

**Turbo LoRAs and Spectrum both cost you audio.** A turbo LoRA hits a
result in very few steps, and fine detail is what those last steps were
for. It thickens the sound and softens the picture. Step-skipping
optimizers like ComfyUI-Spectrum-MiniMax-H3 do the same thing to the
audio, less so to the picture, and they also mispredict the pinned rows,
which never change. Run both together and it stacks. If a chain sounds
duller or closer than you expected, try turning these off before blaming
the chaining. **Keep Spectrum off for these graphs.**

**Resolution can't change mid-chain** while using `context_latent`. A
latent can't be resized, so the node refuses. Regenerate the previous clip
at the new resolution, or start a fresh chain there.

**Prompting a chain takes work**, especially with reference mode. Every
reference conditions the whole clip; there's no way to say "this one
starts two seconds in." Timing has to come from the shot structure and
the description. See "Writing prompts for a chain" above.

**Tested narrowly.** Joins have been verified on dense beat-driven
electronic music, where timing errors are most audible, and on spoken word
through the latent path, where nothing hides a seam. One Windows machine,
one resolution, one sampler. The math self-tests every startup. The
perceptual results are one person's renders.

**ComfyUI's H3 support is young.** These patches depend on the current
shape of it. They check their assumptions at startup and shut down if
something moved, so the failure mode after an update is "the node won't
run," not bad output.

**License.** The H3 community license reportedly doesn't currently cover
the EU, UK, Korea or the US. Check for yourself before shipping anything
on it.

## Recommended starting point

`context_length 22`, `audio_context_length 22`, `context_latent` wired
through the Save/Load Latent pair, Trim node wired for picture and sound
with `match_tail` on, Spectrum off. Every "it works" in this README means
that config.

## Testing

Six scripts, all runnable without ComfyUI or a GPU.

```
python tests/_mock_harness.py        # patch logic against a fake stock model
python tests/_node_smoke_test.py     # the node end to end, refs + save/load
python tests/_payload_gate_test.py   # unrelated H3 graphs come out unchanged
python tests/seam_probe.py A.flac B_untrimmed.flac    # is the join real continuation?
python tests/level_step.py clip*.flac                 # does the level or room tone jump?
python tests/freeze_detect.py clip*.mp4               # did a held shot render as a still?
```

The first three print their checks and end with a pass line. The other three
measure real output and each takes `--self-test` to check its own math on
made-up data first.

The three measurement scripts ask different questions and a join can fail
any one on its own. `seam_probe` is timing: is the audio the same waveform
continued, or a sound-alike. `level_step` is volume: does the loudness, or
the room tone underneath it, jump at the cut. `freeze_detect` is picture:
is anything happening in that shot.

For `seam_probe` and `level_step`, give them the UNTRIMMED audio of the new
clip. Branch the audio decode to a second Save Audio node alongside the
Trim.

## Upgrading

The node's widgets changed. ComfyUI stores widget values by position, so a
workflow saved against an older version will load its numbers into the
wrong slots. Delete the Motion Context node, add it again and rewire it.
Takes a minute and beats rendering with scrambled settings.

**Only install this once**, and don't run it alongside another pack that
does the same job. Several H3 packs lift the same first/last keyframe
restriction independently, and only one of them can own that code. If
another one got there first this node says so and refuses, rather than
wrapping their patch and producing joins neither pack intended. Disable
one and restart.

The same goes for a fork of this repo, a manual clone next to a Manager
install, or a renamed backup still sitting in `custom_nodes`. Renaming a
folder does not stop ComfyUI loading it.

## Credits

The Ref2VA multi-reference support was worked out by **seitanism** in the
Banodoco MiniMax H3 seamless-extension thread and first implemented in
**ethanfel**'s fork of this repo. The code here is written independently,
but the idea is theirs and they got there first.

The prompting section comes out of a 16-clip, 4:34 multicam sitcom episode
chained with this pack at 736x576 on a 5070 Ti, about two hours of
generation. The airlock, the freeze warning, the 32 kHz trap and the
timecode budget are all that build's findings. Same run measured a
video-only chain at a median seam level step of 0.905, dropping to 0.16
with the audio carried across.

If you build something long with this, numbers are more useful than praise.
Open an issue.

## Files

| File | Role |
|---|---|
| `patch_layout.py` | Lifts the first/last-only pinned frame restriction, moves pinned audio onto the clip's timeline, keeps everything lined up when references shift the layout. Self-tests at startup. |
| `patch_payload.py` | Lets pinned video and pinned audio coexist. Stock code let one overwrite the other. Only applies to graphs using this pack. |
| `nodes.py` | The four nodes: Motion Context, Trim, and the latent Save/Load pair. |
| `tests/seam_probe.py` | Is a join's audio a real continuation, a sound-alike, or drifting. |
| `tests/level_step.py` | Level and room-tone continuity at each join. Also catches sample-rate mismatches. |
| `tests/freeze_detect.py` | Stretches where the picture stops moving. |
| `tests/_mock_harness.py`, `tests/_node_smoke_test.py`, `tests/_payload_gate_test.py` | Patch and node tests, numpy only. |
