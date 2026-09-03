"""
Model: https://huggingface.co/Qwen/Qwen3.5-0.8B
Image Count + Resolution: 4 images taken from the spot kinect and resized to 224x336
Prompt: “Describe all of the objects in the scene and their positions in the images in [x,y] coordinates each object’s center of mass”
Inference framework: VLLM

Task 1: Adapt the Dockerfile/container build to be compatible with the Orin and Thor
Task 2: Modify this script to initialize the vLLM server for the Qwen3.5-0.8B, Gemma4-E2B, Qwen3.5-2B model
Task 3: Write a simple prompt to use a context of 4 images (can be from the internet) and ask the model to describe the objects in the scene and their positions in the images in [x,y] coordinates each object’s center of mass
Task 4: Record inference time metrics and throughput for each model and report the results in a table

Metrics:
Time to first token (TFT) and throughput (tokens/sec) for each model on the Orin and Thor platforms. 

Peak memory usage
Power consumption

Keep model sampling parameter consistent within the same model family.
Choose the optimal parameters for each specific model family.

"""