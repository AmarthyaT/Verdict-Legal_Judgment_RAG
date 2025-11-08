import os
import sys
import argparse
from llama_cpp import Llama

def main(args):
    # --- 1. Check if the model file exists ---
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)

    # --- 2. Initialize the Llama model with GPU offloading ---
    print(f"Loading GGUF model from: {args.model}")
    try:
        llm = Llama(
            model_path=args.model,
            n_gpu_layers=-1,  # Offload all layers to GPU. Set to 0 for CPU-only.
            n_ctx=2048,       # The context window size
            verbose=False     # Suppress verbose output from llama.cpp
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # --- 3. Generate text ---
    print("Generating text (this may take a moment)...")
    
    response = llm(
        args.prompt,
        max_tokens=args.max_tokens,
        echo=True  # Echo the prompt in the output
    )

    # --- 4. Print the generated text ---
    # The response object is a dictionary, and the text is in choices[0]['text']
    print("\n--- Generated Response ---")
    print(response['choices'][0]['text'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a GGUF model using llama-cpp-python")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the GGUF model file."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time,",
        help="The prompt to generate text from."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="The maximum number of tokens to generate."
    )
    
    args = parser.parse_args()
    main(args)