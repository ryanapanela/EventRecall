import argparse
import os
from .segmentation import run_segmentation
from .recall import run_recall_scoring

def main():
    parser = argparse.ArgumentParser(description="Automated Event Segmentation and Recall Scoring Tool")
    parser.add_argument('--segmentation', type=str, help='Path to segmentation input file (CSV or TXT)')
    parser.add_argument('--recall', type=str, help='Path to recall input file (CSV or TXT)')
    parser.add_argument('--api_key', type=str, help='OpenAI API key (or set OPENAI_API_KEY env variable)')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OpenAI API key must be provided via --api_key or OPENAI_API_KEY environment variable.')

    os.makedirs(args.output, exist_ok=True)

    if args.segmentation:
        run_segmentation(args.segmentation, api_key, args.output)
    if args.recall:
        run_recall_scoring(args.recall, api_key, args.output)

if __name__ == "__main__":
    main()
