try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError as e:
    # print("Cannot import vllm")
    pass

from lcb_runner.runner.base_runner import BaseRunner


class VLLMRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        model_tokenizer_path = (
            model.model_name if args.local_model_path is None else args.local_model_path
        )
        print("Loading VLLM model from", model_tokenizer_path)
        self.llm = LLM(
            model=model_tokenizer_path,
            tokenizer=model_tokenizer_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            enforce_eager=False,
            disable_custom_all_reduce=True,
            enable_prefix_caching=args.enable_prefix_caching,
            trust_remote_code=args.trust_remote_code,
        )
        # print all info
        print("Model name", model_tokenizer_path)
        print("Tensor parallel size", args.tensor_parallel_size)
        print("Dtype", args.dtype)
        print("Enable prefix caching", args.enable_prefix_caching)
        print("Trust remote code", args.trust_remote_code)
        print("Max token", self.args.max_tokens)
        self.sampling_params = SamplingParams(
            n=self.args.n,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=self.args.stop,
        )
        print("Sampling params", self.sampling_params)
        

    def _run_single(self, prompt: str) -> list[str]:
        pass

    def run_batch(self, prompts: list[str]) -> list[list[str]]:
        outputs = [None for _ in prompts]
        remaining_prompts = []
        remaining_indices = []
        for prompt_index, prompt in enumerate(prompts):
            if self.args.use_cache and prompt in self.cache:
                if len(self.cache[prompt]) == self.args.n:
                    outputs[prompt_index] = self.cache[prompt]
                    continue
            remaining_prompts.append(prompt)
            remaining_indices.append(prompt_index)
        print("Remaining prompts", remaining_prompts)
        print("that is remaining prompt")
        if remaining_prompts:
            print( "prompts remaining", len(remaining_prompts))
            print("Running VLLM model for remaining prompts")
            vllm_outputs = self.llm.generate(remaining_prompts, self.sampling_params)
            #print("VLLM outputs", vllm_outputs)
            if self.args.use_cache:
                assert len(remaining_prompts) == len(vllm_outputs)
                for index, remaining_prompt, vllm_output in zip(
                    remaining_indices, remaining_prompts, vllm_outputs
                ):
                    self.cache[remaining_prompt] = [o.text for o in vllm_output.outputs]
                    outputs[index] = [o.text for o in vllm_output.outputs]
            else:
                for index, vllm_output in zip(remaining_indices, vllm_outputs):
                    outputs[index] = [o.text for o in vllm_output.outputs]
            """batch_size = 5
            all_outputs = []
            
            for i in range(0, len(remaining_prompts), batch_size):
                batch_prompts = remaining_prompts[i:i+batch_size]
                batch_indices = remaining_indices[i:i+batch_size]
                
                print(f"Processing batch {i//batch_size + 1}/{(len(remaining_prompts) + batch_size - 1)//batch_size}: {len(batch_prompts)} prompts")
                vllm_outputs = self.llm.generate(batch_prompts, self.sampling_params)
                print(f"Completed batch {i//batch_size + 1}")
                print("VLLM outputs", vllm_outputs)
                
                if self.args.use_cache:
                    assert len(batch_prompts) == len(vllm_outputs)
                    for index, batch_prompt, vllm_output in zip(
                        batch_indices, batch_prompts, vllm_outputs
                    ):
                        self.cache[batch_prompt] = [o.text for o in vllm_output.outputs]
                        outputs[index] = [o.text for o in vllm_output.outputs]
                else:
                    for index, vllm_output in zip(batch_indices, vllm_outputs):
                        outputs[index] = [o.text for o in vllm_output.outputs]"""
        return outputs
