"""Run the looping orchestration end to end without invoking the DiT."""

import importlib.util
import json
import os
import sys
import tempfile

import torch

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_DIR = os.path.dirname(os.path.dirname(PACKAGE_DIR))
sys.path.insert(0, COMFY_DIR)

import comfy.cli_args

comfy.cli_args.args.cpu = True

import comfy.nested_tensor  # noqa: E402


def main():
    spec = importlib.util.spec_from_file_location(
        "h3mc_loop_test", os.path.join(PACKAGE_DIR, "__init__.py"),
        submodule_search_locations=[PACKAGE_DIR])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    looping = sys.modules["h3mc_loop_test.looping_sampler"]

    schema_info = looping.MiniMaxH3LoopingSampler.GET_NODE_INFO_V1()
    assert schema_info["input"]["optional"]["tile_latents"][1][
        "display_name"] == "Tile latents (group)"
    assert schema_info["input"]["optional"]["tile_conditionings"][1][
        "display_name"] == "Tile conditionings (group)"

    with open(os.path.join(PACKAGE_DIR, "example_workflows",
                           "minimax_h3_looping.json"), encoding="utf-8") as file:
        workflow = json.load(file)
    sampler_node = next(node for node in workflow["nodes"]
                        if node["type"] == "MiniMaxH3LoopingSampler")
    input_names = {input["name"] for input in sampler_node["inputs"]}
    assert "tile_latents.tile_latent_0" in input_names
    assert "tile_conditionings.tile_conditioning_0" in input_names
    assert "tile_latent_0" not in input_names
    assert "tile_conditioning_0" not in input_names

    def make_latent(video_t, audio_t):
        return {"samples": comfy.nested_tensor.NestedTensor([
            torch.zeros((1, 24, video_t, 2, 2)),
            torch.zeros((1, 32, 2, audio_t)),
        ])}

    latent = make_latent(37, 207)
    tile_latents = {
        "tile_latent_0": latent,
        "tile_latent_1": make_latent(42, 235),
        "tile_latent_2": make_latent(47, 263),
    }
    assert looping._autogrow_values(
        {"tile_conditioning_0": "first", "tile_conditioning_2": "third"},
        "tile_conditioning_") == {0: "first", 2: "third"}

    calls = []

    def fake_sample(model, conditioning, sampler, sigmas, target, seed):
        calls.append((seed, conditioning))
        out = looping._copy_latent(target)
        value = float(len(calls))
        for stream in out["samples"].unbind():
            stream.fill_(value)
        return out

    looping._sample_tile = fake_sample

    class Clip:
        def tokenize(self, prompt):
            return prompt

        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.zeros((1, 1)), {"prompt": tokens}]]

    prompt_result = looping.MiniMaxH3MultiPromptProvider.execute(
        clip=Clip(), global_prompt="A silver station wagon at dawn.",
        tile_prompts="Wide road shot.|The wagon passes a guardrail.|The road bends toward the horizon.")
    assert [entry[0][1]["prompt"] for entry in prompt_result.args[0]] == [
        "A silver station wagon at dawn.\n\nWide road shot.",
        "A silver station wagon at dawn.\n\nThe wagon passes a guardrail.",
        "A silver station wagon at dawn.\n\nThe road bends toward the horizon.",
    ]

    decode_calls = []

    class VideoVAE:
        def decode(self, video, vae_options=None):
            decode_calls.append(vae_options)
            value = float(video[0, 0, 0, 0, 0])
            frame_count = looping._pixel_frames(video.shape[2])
            return torch.full((1, frame_count, 32, 32, 3), value)

    class AudioVAE:
        audio_sample_rate_output = 32000

        def decode(self, audio):
            value = float(audio[0, 0, 0, 0])
            frame_count = {207: 124, 235: 141, 263: 158}[audio.shape[-1]]
            samples = round(frame_count / 24 * self.audio_sample_rate_output) + 8
            return torch.full((1, samples, 2), value)

    result = looping.MiniMaxH3LoopingSampler.execute(
        model=object(), positive=[[torch.zeros((1, 1)), {}]],
        vae=VideoVAE(), audio_vae=AudioVAE(), sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]), latent=latent, tiles=3,
        context_frames="22", seed=100,
        prompt_conditionings=prompt_result.args[0], tile_latents=tile_latents,
        decode_mode="advanced", decode_tile_size="512",
        checkpoint_prefix="")
    images, audio, last_latent, delivered_frames = result.args

    assert [seed for seed, _ in calls] == [100, 101, 102]
    assert delivered_frames == 124 + (141 - 22) + (158 - 22) == 379
    assert images.shape == (379, 32, 32, 3)
    assert audio["sample_rate"] == 32000
    assert audio["waveform"].shape[-1] == round(379 / 24 * 32000)
    assert last_latent["samples"].unbind()[0].device.type == "cpu"
    assert decode_calls == [{
        "decode_options": {"tile_size": 512, "tile_overlap": 128},
    }] * 3

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        original_output_directory = looping.folder_paths.get_output_directory
        looping.folder_paths.get_output_directory = lambda: checkpoint_dir
        try:
            path = looping._save_tile_checkpoint(
                last_latent, "h3_checkpoint", 3)
            loaded, tile_number, loaded_path = (
                looping.MiniMaxH3LoopingSamplerLoadCheckpoint.execute(
                    "h3_checkpoint").args)
            assert tile_number == 3
            assert loaded_path == path
            assert loaded["samples"].unbind()[0].device.type == "cpu"
            for expected, actual in zip(last_latent["samples"].unbind(),
                                        loaded["samples"].unbind()):
                assert torch.equal(expected, actual)
        finally:
            looping.folder_paths.get_output_directory = original_output_directory
    print("checkpoint recovery: newest completed AV tile loaded for decoding")

    class BrokenVideoVAE(VideoVAE):
        def decode(self, video, vae_options=None):
            raise RuntimeError("synthetic decode failure")

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        original_output_directory = looping.folder_paths.get_output_directory
        looping.folder_paths.get_output_directory = lambda: checkpoint_dir
        try:
            try:
                looping.MiniMaxH3LoopingSampler.execute(
                    model=object(), positive=[[torch.zeros((1, 1)), {}]],
                    vae=BrokenVideoVAE(), audio_vae=AudioVAE(),
                    sampler=object(), sigmas=torch.tensor([1.0, 0.0]),
                    latent=latent, tiles=3, context_frames="22", seed=100,
                    prompt_conditionings=prompt_result.args[0],
                    tile_latents=tile_latents, decode_mode="advanced",
                    decode_tile_size="512", checkpoint_prefix="failed_run")
            except RuntimeError as error:
                assert "synthetic decode failure" in str(error)
            else:
                raise AssertionError("synthetic decode failure was swallowed")
            _, tile_number, _ = (
                looping.MiniMaxH3LoopingSamplerLoadCheckpoint.execute(
                    "failed_run").args)
            assert tile_number == 3
        finally:
            looping.folder_paths.get_output_directory = original_output_directory
    print("failure recovery: tile 3 checkpoint survived a decode exception")

    class MemoryVAE:
        upscale_ratio = (None, 16, 16)
        vae_dtype = torch.float16

        class Patcher:
            def get_free_memory(self, device):
                return 1_000_000_000

            def model_size(self):
                return 0

        patcher = Patcher()
        device = torch.device("cpu")

        def memory_used_decode(self, shape, dtype):
            return shape[-1] ** 2 * 100_000

    auto_options = looping._resolve_decode_options(
        MemoryVAE(), torch.zeros((1, 24, 7, 1, 1)), "auto", "512")
    assert auto_options == {"tile_size": 1024, "tile_overlap": 256}

    try:
        looping.MiniMaxH3LoopingSampler.execute(
            model=object(), positive=[[torch.zeros((1, 1)), {}]],
            vae=VideoVAE(), audio_vae=AudioVAE(), sampler=object(),
            sigmas=torch.tensor([1.0, 0.0]), latent=latent, tiles=3,
            context_frames="22", seed=100, settling_tail_frames=1,
            checkpoint_prefix="")
    except ValueError as error:
        assert "multiple of 17" in str(error)
    else:
        raise AssertionError("off-grid settling tail was accepted")

    indexed = {
        "samples": comfy.nested_tensor.NestedTensor([
            torch.arange(1 * 24 * 37 * 2 * 2).reshape(1, 24, 37, 2, 2),
            torch.arange(1 * 32 * 2 * 207).reshape(1, 32, 2, 207),
        ])
    }
    prefix, video_steps, audio_steps = looping._slice_latent_prefix(
        indexed, 107)
    assert (video_steps, audio_steps) == (32, 178)
    assert torch.equal(prefix["samples"].unbind()[0],
                       indexed["samples"].unbind()[0][:, :, :32])
    assert torch.equal(prefix["samples"].unbind()[1],
                       indexed["samples"].unbind()[1][:, :, :, :178])
    print("settling trim: 107 frames selects video step 32 and audio step 178")

    first_values = calls[0][1][0][1]
    second_values = calls[1][1][0][1]
    assert first_values["prompt"].endswith("Wide road shot.")
    assert second_values["prompt"].endswith("The wagon passes a guardrail.")
    assert "minimax_keyframes" not in first_values
    assert len(second_values["minimax_keyframes"]) == 7
    assert len(second_values["minimax_refs"]) == 1
    assert second_values["minimax_refs"][0]["kind"] == "audio"

    def anchored(frame_count):
        return [["c", {
            "minimax_frame_count": frame_count,
            "minimax_keyframes": [
                {"resolved_frame_index": 0},
                {"resolved_frame_index": frame_count - 1},
            ],
        }]]

    def unanchored(frame_count):
        return [["c", {"minimax_frame_count": frame_count}]]

    original_context_apply = looping.MiniMaxH3MotionContext.apply
    context_latents = []

    def fake_context(self, conditioning, vae, latent, context_length,
                     audio_context_length=0, context_latent=None, **kwargs):
        context_latents.append(context_latent)
        return conditioning, int(context_length)

    looping.MiniMaxH3MotionContext.apply = fake_context
    calls.clear()
    context_latents.clear()
    anchored_result = looping.MiniMaxH3LoopingSampler.execute(
        model=object(), positive=unanchored(124), vae=VideoVAE(),
        audio_vae=AudioVAE(), sampler=object(), sigmas=torch.tensor([1.0, 0.0]),
        latent=latent, tiles=3, context_frames="22", seed=100,
        tile_latents=tile_latents,
        tile_conditionings={
            "tile_conditioning_0": anchored(124),
            "tile_conditioning_1": anchored(141),
            "tile_conditioning_2": anchored(158),
        }, settling_tail_frames=17, checkpoint_prefix="")
    _, anchored_audio, _, anchored_frames = anchored_result.args
    assert anchored_frames == (124 - 17) + (141 - 22 - 17) + (158 - 22)
    assert anchored_audio["waveform"].shape[-1] == round(
        anchored_frames / 24 * 32000)
    assert [tuple(item["samples"].unbind()[0].shape[2:])
            for item in context_latents] == [(32, 2, 2), (37, 2, 2)]
    assert [item["samples"].unbind()[1].shape[-1]
            for item in context_latents] == [178, 207]
    print("anchored intermediate tiles: endpoint tail removed, final tile kept")

    calls.clear()
    context_latents.clear()
    plain_result = looping.MiniMaxH3LoopingSampler.execute(
        model=object(), positive=unanchored(124), vae=VideoVAE(),
        audio_vae=AudioVAE(), sampler=object(), sigmas=torch.tensor([1.0, 0.0]),
        latent=latent, tiles=3, context_frames="22", seed=100,
        tile_latents=tile_latents,
        tile_conditionings={
            "tile_conditioning_0": unanchored(124),
            "tile_conditioning_1": unanchored(141),
            "tile_conditioning_2": anchored(158),
        }, settling_tail_frames=17, checkpoint_prefix="")
    _, plain_audio, _, plain_frames = plain_result.args
    assert plain_frames == 379
    assert plain_audio["waveform"].shape[-1] == round(379 / 24 * 32000)
    assert [tuple(item["samples"].unbind()[0].shape[2:])
            for item in context_latents] == [(37, 2, 2), (42, 2, 2)]
    looping.MiniMaxH3MotionContext.apply = original_context_apply
    print("unanchored intermediate tiles: settling trim left disabled")
    print("looping sampler: 3 tiles, deterministic seeds, motion context, "
          "and exact AV duration passed")


if __name__ == "__main__":
    main()
