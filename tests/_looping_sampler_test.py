"""Run the looping orchestration end to end without invoking the DiT."""

import importlib.util
import os
import sys

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

    class VideoVAE:
        def decode(self, video):
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
        prompt_conditionings=prompt_result.args[0], tile_latents=tile_latents)
    images, audio, last_latent, delivered_frames = result.args

    assert [seed for seed, _ in calls] == [100, 101, 102]
    assert delivered_frames == 124 + (141 - 22) + (158 - 22) == 379
    assert images.shape == (379, 32, 32, 3)
    assert audio["sample_rate"] == 32000
    assert audio["waveform"].shape[-1] == round(379 / 24 * 32000)
    assert last_latent["samples"].unbind()[0].device.type == "cpu"

    first_values = calls[0][1][0][1]
    second_values = calls[1][1][0][1]
    assert first_values["prompt"].endswith("Wide road shot.")
    assert second_values["prompt"].endswith("The wagon passes a guardrail.")
    assert "minimax_keyframes" not in first_values
    assert len(second_values["minimax_keyframes"]) == 7
    assert len(second_values["minimax_refs"]) == 1
    assert second_values["minimax_refs"][0]["kind"] == "audio"
    print("looping sampler: 3 tiles, deterministic seeds, motion context, "
          "and exact AV duration passed")


if __name__ == "__main__":
    main()
