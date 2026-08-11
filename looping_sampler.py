"""Temporal tiled sampling for long MiniMax H3 audio-video generations."""

import logging

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.utils
import latent_preview
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import Guider_Basic

from .nodes import (
    FPS,
    MC_KEY,
    MiniMaxH3MotionContext,
    _ensure_layout_patch,
    _ensure_payload_patch,
    _pixel_frames,
    _streams_from_latent,
)

_LOG = logging.getLogger("h3_motion_context")


def _autogrow_values(values, prefix):
    values = values or {}
    return [values[name] for name in sorted(
        values, key=lambda name: int(name.removeprefix(prefix)))]


def _copy_latent(latent, device=None):
    parts = _streams_from_latent(latent)
    if len(parts) != 2:
        raise ValueError(
            "h3_motion_context: the looping sampler needs an H3 joint "
            "video/audio latent")
    copied = latent.copy()
    if device is None:
        tensors = [part.clone() for part in parts]
    else:
        tensors = [part.to(device).clone() for part in parts]
    copied["samples"] = comfy.nested_tensor.NestedTensor(tensors)
    return copied


def _keyframe_position(keyframe):
    p = keyframe.get(MC_KEY, keyframe.get("resolved_frame_index"))
    if p is None:
        raise ValueError(
            "h3_motion_context: a keyframe has no resolved frame index")
    return int(p)


def _conditioning_for_tile(conditioning, tile_index, tiles, frame_count,
                           tile_specific):
    """Keep only the endpoints that belong to this temporal tile."""
    out = []
    has_keyframes_and_refs = False
    for entry in conditioning:
        copied = list(entry)
        values = entry[1].copy()
        source_frame_count = int(values.get("minimax_frame_count", frame_count))
        keyframes = []
        for keyframe in values.get("minimax_keyframes", []):
            p = _keyframe_position(keyframe)
            if p == 0:
                keep = tile_index == 0
                target = 0
            elif p == source_frame_count - 1:
                keep = tile_specific or tile_index == tiles - 1
                target = frame_count - 1
            else:
                raise ValueError(
                    "h3_motion_context: looping inputs may contain only "
                    "first/last FL2VA anchors; got frame %d of %d"
                    % (p, source_frame_count))
            if keep:
                kept = keyframe.copy()
                kept["resolved_frame_index"] = 0
                kept[MC_KEY] = target
                keyframes.append(kept)

        if keyframes:
            values["minimax_keyframes"] = keyframes
            values["minimax_frame_count"] = frame_count
            has_keyframes_and_refs |= bool(values.get("minimax_refs"))
        else:
            values.pop("minimax_keyframes", None)
            values.pop("minimax_frame_count", None)
        copied[1] = values
        out.append(copied)

    if any(entry[1].get("minimax_keyframes") for entry in out):
        _ensure_layout_patch()
    if has_keyframes_and_refs:
        _ensure_payload_patch()
    return out


def _sample_tile(model, conditioning, sampler, sigmas, latent, seed):
    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent["samples"], latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"))
    sampled_latent = latent.copy()
    sampled_latent["samples"] = latent_image
    batch_inds = latent.get("batch_index")
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    guider = Guider_Basic(model)
    guider.set_conds(conditioning)
    callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)
    samples = guider.sample(
        noise, latent_image, sampler, sigmas,
        denoise_mask=latent.get("noise_mask"), callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    sampled_latent.pop("downscale_ratio_spacial", None)
    sampled_latent.pop("downscale_ratio_temporal", None)
    sampled_latent["samples"] = samples.to(
        comfy.model_management.intermediate_device())
    return sampled_latent


def _decode_video(vae, latent):
    video = _streams_from_latent(latent)[0]
    images = vae.decode(video)
    if images.ndim == 5:
        images = images.reshape(
            -1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def _decode_audio(audio_vae, latent):
    audio = _streams_from_latent(latent)[1]
    waveform = audio_vae.decode(audio).movedim(-1, 1)
    sample_rate = int(getattr(
        audio_vae, "audio_sample_rate_output",
        getattr(audio_vae, "audio_sample_rate", 32000)))
    return waveform, sample_rate


def _trim_decoded(images, waveform, sample_rate, trim_frames, frame_count,
                  delivered_start, delivered_end):
    if images.shape[0] < frame_count:
        raise ValueError(
            "h3_motion_context: video VAE decoded %d frames, expected %d"
            % (images.shape[0], frame_count))
    images = images[trim_frames:frame_count]
    start = round(trim_frames / FPS * sample_rate)
    want = (round(delivered_end / FPS * sample_rate)
            - round(delivered_start / FPS * sample_rate))
    if waveform.shape[-1] < start + want:
        raise ValueError(
            "h3_motion_context: audio VAE decoded %.4fs, but this tile needs "
            "%.4fs after its context head"
            % (waveform.shape[-1] / sample_rate,
               frame_count / FPS))
    return images.cpu(), waveform[..., start:start + want].cpu()


class MiniMaxH3MultiPromptProvider(io.ComfyNode):
    """Encode a shared prompt plus one standalone prompt per tile."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MultiPromptProvider",
            display_name="MiniMax H3 Multi-Prompt Provider",
            category="conditioning/minimax",
            description=(
                "Encode one standalone H3 prompt per tile, prepending the "
                "same global prompt to every tile."),
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input(
                    "global_prompt", multiline=True, dynamic_prompts=True,
                    optional=True, default="",
                    tooltip="Text shared by every tile."),
                io.String.Input(
                    "tile_prompts", multiline=True, dynamic_prompts=True,
                    default="",
                    tooltip="One standalone prompt per tile, separated by |."),
            ],
            outputs=[io.Conditioning.Output(display_name="conditionings")],
        )

    @classmethod
    def execute(cls, clip, global_prompt, tile_prompts):
        global_prompt = global_prompt.strip()
        prompts = [prompt.strip() for prompt in tile_prompts.split("|")]
        if not prompts or all(not prompt for prompt in prompts):
            prompts = [""]

        encoded = []
        for prompt in prompts:
            full_prompt = "\n\n".join(
                part for part in (global_prompt, prompt) if part)
            encoded.append(clip.encode_from_tokens_scheduled(
                clip.tokenize(full_prompt)))
        return io.NodeOutput(encoded)


class MiniMaxH3LoopingSampler(io.ComfyNode):
    """Sample H3 windows sequentially and carry their joint AV latent tail."""

    @classmethod
    def define_schema(cls):
        conditioning_template = io.Autogrow.TemplatePrefix(
            input=io.Conditioning.Input(
                "tile_conditioning",
                tooltip="Optional prompt/FL2VA endpoints for one tile."),
            prefix="tile_conditioning_", min=0, max=64)
        return io.Schema(
            node_id="MiniMaxH3LoopingSampler",
            display_name="MiniMax H3 Looping Sampler",
            category="sampling/minimax",
            description=(
                "Generate a long H3 clip as temporal tiles. Every later tile "
                "pins and discards the previous tile's joint video/audio "
                "latent tail."),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input(
                    "positive",
                    tooltip="Default conditioning. Its first anchor belongs "
                            "to tile 1 and its last anchor to the final tile."),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input(
                    "latent",
                    tooltip="One empty H3 AV tile. Its legal 17k+5 frame "
                            "length and resolution define every tile."),
                io.Int.Input("tiles", default=3, min=1, max=64),
                io.Combo.Input("context_frames", options=["22", "5", "39", "56"], default="22"),
                io.Int.Input("seed", default=0, min=0,
                             max=0xffffffffffffffff,
                             control_after_generate=True),
                io.Conditioning.Input(
                    "prompt_conditionings",
                    optional=True,
                    tooltip=(
                        "Prompt list from MiniMax H3 Multi-Prompt Provider. "
                        "The final item repeats if fewer prompts than tiles "
                        "are supplied.")),
                io.Autogrow.Input(
                    "tile_conditionings", template=conditioning_template,
                    optional=True,
                    tooltip="Overrides the default conditioning by tile. "
                            "Use stock H3 conditioning nodes so prompt image "
                            "tokens and FL2VA end anchors stay intact."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.Latent.Output(display_name="last_tile_latent"),
                io.Int.Output(display_name="frame_count"),
            ],
        )

    @classmethod
    def execute(cls, model, positive, vae, audio_vae, sampler, sigmas,
                latent, tiles, context_frames, seed,
                prompt_conditionings=None, tile_conditionings=None):
        tiles = int(tiles)
        context_frames = int(context_frames)
        parts = _streams_from_latent(latent)
        if len(parts) != 2 or parts[0].ndim != 5 or parts[1].ndim != 4:
            raise ValueError(
                "h3_motion_context: latent is not an H3 joint AV tile")
        if parts[0].shape[0] != 1 or parts[1].shape[0] != 1:
            raise ValueError(
                "h3_motion_context: looping currently supports batch size 1")
        frame_count = _pixel_frames(int(parts[0].shape[2]))
        if context_frames >= frame_count:
            raise ValueError(
                "h3_motion_context: context_frames must be shorter than the "
                "tile (%d frames)" % frame_count)

        overrides = _autogrow_values(
            tile_conditionings, "tile_conditioning_")
        prompt_conditionings = prompt_conditionings or []
        if len(overrides) > tiles:
            raise ValueError(
                "h3_motion_context: %d tile conditionings supplied for %d tiles"
                % (len(overrides), tiles))

        sampled_tiles = []
        previous = None
        for tile_index in range(tiles):
            comfy.model_management.throw_exception_if_processing_interrupted()
            if tile_index < len(overrides):
                source = overrides[tile_index]
                tile_specific = True
            elif prompt_conditionings:
                source = prompt_conditionings[min(
                    tile_index, len(prompt_conditionings) - 1)]
                tile_specific = True
            else:
                source = positive
                tile_specific = False
            conditioning = _conditioning_for_tile(
                source, tile_index, tiles, frame_count, tile_specific)
            target = _copy_latent(latent)
            trim_frames = 0
            if previous is not None:
                conditioning, trim_frames = MiniMaxH3MotionContext().apply(
                    conditioning=conditioning, vae=vae, latent=target,
                    context_length=str(context_frames),
                    audio_context_length=0, context_latent=previous)
            tile_seed = (int(seed) + tile_index) & 0xffffffffffffffff
            _LOG.info(
                "h3_motion_context: sampling tile %d/%d, %d frames, trim %d, "
                "seed %d", tile_index + 1, tiles, frame_count, trim_frames,
                tile_seed)
            sampled = _sample_tile(
                model, conditioning, sampler, sigmas, target, tile_seed)
            previous = _copy_latent(sampled, device="cpu")
            sampled_tiles.append((previous, trim_frames))

        image_chunks = []
        audio_chunks = []
        sample_rate = None
        delivered_start = 0
        for tile_index, (sampled, trim_frames) in enumerate(sampled_tiles):
            comfy.model_management.throw_exception_if_processing_interrupted()
            images = _decode_video(vae, sampled)
            waveform, tile_sample_rate = _decode_audio(audio_vae, sampled)
            if sample_rate is None:
                sample_rate = tile_sample_rate
            elif sample_rate != tile_sample_rate:
                raise ValueError(
                    "h3_motion_context: audio sample rate changed between tiles")
            delivered_end = delivered_start + frame_count - trim_frames
            images, waveform = _trim_decoded(
                images, waveform, sample_rate, trim_frames, frame_count,
                delivered_start, delivered_end)
            image_chunks.append(images)
            audio_chunks.append(waveform)
            delivered_start = delivered_end
            _LOG.info(
                "h3_motion_context: decoded tile %d/%d -> %d delivered frames",
                tile_index + 1, tiles, images.shape[0])

        images = torch.cat(image_chunks, dim=0)
        waveform = torch.cat(audio_chunks, dim=-1)
        std = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        waveform /= std
        delivered_frames = frame_count + (tiles - 1) * (
            frame_count - context_frames)
        if images.shape[0] != delivered_frames:
            raise RuntimeError(
                "h3_motion_context: assembled %d frames, planned %d"
                % (images.shape[0], delivered_frames))
        expected_samples = round(delivered_frames / FPS * sample_rate)
        if waveform.shape[-1] != expected_samples:
            raise RuntimeError(
                "h3_motion_context: assembled %d audio samples, planned %d"
                % (waveform.shape[-1], expected_samples))

        return io.NodeOutput(
            images, {"waveform": waveform, "sample_rate": sample_rate},
            sampled_tiles[-1][0], delivered_frames)
