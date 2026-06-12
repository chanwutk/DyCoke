"""lmms-eval model wrapper for Qwen3.5 (natively multimodal).

Supports video benchmarks (VideoMME, MVBench, LongVideoBench) using
Qwen3.5's built-in vision encoder and processor.

Usage in lmms-eval:
    --model qwen3_5
    --model_args pretrained=Qwen/Qwen3.5-9B,max_frames_num=32
"""

import logging
import os
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")
eval_logger = logging.getLogger("lmms-eval")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = False


@register_model("qwen3_5")
class Qwen3_5(lmms):
    """Qwen3.5 natively multimodal model wrapper for lmms-eval."""

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-9B",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        device_map: Optional[str] = "cuda:0",
        use_cache: Optional[bool] = True,
        max_frames_num: Optional[int] = None,
        max_pixels: Optional[int] = None,
        attn_implementation: Optional[str] = "eager",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.max_frames_num = max_frames_num
        self.use_cache = use_cache
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "Qwen3.5 wrapper currently supports batch_size=1 only."

        eval_logger.info(f"Loading Qwen3.5 model: {pretrained}")
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            attn_implementation=attn_implementation,
        )
        self._processor = AutoProcessor.from_pretrained(pretrained)

        # Override max_frames if specified (fixed-frames strategy).
        # When None, the processor uses its native FPS-based sampling.
        if self.max_frames_num is not None:
            self._processor.video_processor.max_frames = self.max_frames_num
            self._processor.video_processor.min_frames = min(
                self._processor.video_processor.min_frames, self.max_frames_num
            )
            eval_logger.info(f"Video processor max_frames overridden to {self.max_frames_num}")

        # Override max_pixels to cap per-frame resolution.
        # Qwen3.5 defaults to near-native resolution (~880 tokens/group at 720p).
        # Setting max_pixels=147456 (384*384) matches LLaVA-OV's spatial budget.
        if max_pixels is not None:
            self._processor.image_processor.max_pixels = max_pixels
            self._processor.image_processor.min_pixels = min(
                self._processor.image_processor.min_pixels, max_pixels
            )
            eval_logger.info(f"Image processor max_pixels overridden to {max_pixels}")

        self._model.eval()
        self._config = self._model.config
        self._tokenizer = self._processor.tokenizer

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED
            ], "Unsupported distributed type."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
            if accelerator.distributed_type in (DistributedType.FSDP, DistributedType.DEEPSPEED):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def processor(self):
        return self._processor

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return getattr(self._config, "max_position_embeddings", 32768)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except Exception:
            return self.tokenizer.decode([tokens])

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _build_video_content(self, visual):
        """Build the video content item for a Qwen3.5 message.

        Handles both video file paths and frame directories (e.g. TVQA
        episodic_reasoning in MVBench, where frames are stored as JPEGs).
        """
        if visual is None or visual == []:
            return None

        path = visual[0] if isinstance(visual, list) else visual

        if isinstance(path, str):
            if os.path.isdir(path):
                # Frame directory — load as list of PIL images.
                frame_files = sorted(
                    f for f in os.listdir(path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                )
                if self.max_frames_num is not None and len(frame_files) > self.max_frames_num:
                    indices = np.linspace(0, len(frame_files) - 1, self.max_frames_num, dtype=int)
                    frame_files = [frame_files[i] for i in indices]
                frames = [Image.open(os.path.join(path, f)).convert("RGB") for f in frame_files]
                return {"type": "video", "video": frames}
            else:
                # Load video via decord (torchvision.io.read_video unavailable).
                from decord import VideoReader, cpu
                vr = VideoReader(path, ctx=cpu(0))
                total = len(vr)
                n = self.max_frames_num or total
                indices = np.linspace(0, total - 1, min(n, total), dtype=int)
                frames_np = vr.get_batch(indices.tolist()).asnumpy()
                frames = [Image.fromarray(f) for f in frames_np]
                return {"type": "video", "video": frames}
        elif isinstance(path, Image.Image):
            # List of PIL images treated as video frames.
            frames = list(visual) if isinstance(visual, list) else [visual]
            return {"type": "video", "video": frames}

        eval_logger.warning(f"Unsupported visual type: {type(path)}")
        return None

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]
            assert len(batched_visuals) == 1

            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            context = batched_contexts[0]
            visual = batched_visuals[0]

            # Build Qwen3.5 message format.
            content = []
            video_content = self._build_video_content(visual)
            if video_content is not None:
                content.append(video_content)
            content.append({"type": "text", "text": context})

            messages = [{"role": "user", "content": content}]

            # Configure generation params.
            max_new_tokens = gen_kwargs.get("max_new_tokens", 1024)
            temperature = gen_kwargs.get("temperature", 0)
            do_sample = gen_kwargs.get("do_sample", False)
            top_p = gen_kwargs.get("top_p", None)
            num_beams = gen_kwargs.get("num_beams", 1)

            try:
                # Processor handles tokenization, video loading/preprocessing,
                # and video token placeholder insertion.
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                ).to(self.device)

                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        use_cache=self.use_cache,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        top_p=top_p,
                        num_beams=num_beams,
                    )

                # Trim input tokens from output.
                input_len = inputs["input_ids"].shape[1]
                output_ids = generated_ids[:, input_len:]
                text_output = self.processor.batch_decode(
                    output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

            except Exception as e:
                eval_logger.error(f"Error during generation: {e}")
                text_output = ""

            res.append(text_output)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_output)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood not implemented for Qwen3.5 wrapper")
