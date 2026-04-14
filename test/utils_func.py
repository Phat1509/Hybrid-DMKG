import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="In Context Learning MH Model Editing")
    parser.add_argument('--edit', type=int, default=1)
    parser.add_argument('--topK', type=int, default=5)
    parser.add_argument("--seed", type=int, default=97, help="Random seed for reproducibility")
    parser.add_argument('--model_name',type=str, default='BLIP2-OPT')
    parser.add_argument('--divide_modal',type=str, default="api-gpt-3.5-turbo")
    parser.add_argument('--image_type', type=int, default=100)
    parser.add_argument('--port', type=int, default=5005, help="Port for the API server")
    parser.add_argument('--information_type', type=str, help="image or image+text")
    args = parser.parse_args()

    return args