import argparse
import os
import sys

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args
from genrec.recommender import Recommender

def parse_args():
    parser = argparse.ArgumentParser(description='RPPG Recommendation System')

    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'generate'],
                        help='Operation mode: train, evaluate, or generate recommendations')

    # Model and dataset arguments
    parser.add_argument('--model', type=str, default='RPG', help='Model name')
    parser.add_argument('--dataset', type=str, default='Pixel', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')

    # Recommendation generation arguments
    parser.add_argument('--output', type=str, default='recommendations.json',
                        help='Output JSON file for recommendations (generate mode only)')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of recommendations per user (generate mode only)')
    parser.add_argument('--user_list', type=str, default=None,
                        help='File containing list of user IDs to generate recommendations for (one per line)')
    parser.add_argument('--include_scores', action='store_true',
                        help='Include confidence scores in output (generate mode only)')
    parser.add_argument('--use_graph_decoding', action='store_true',
                        help='Use graph-constrained decoding (generate mode only)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for recommendation generation (generate mode only)')

    return parser.parse_known_args()

def load_user_list(user_list_file):
    """Load user IDs from file."""
    if not os.path.exists(user_list_file):
        raise FileNotFoundError(f"User list file not found: {user_list_file}")

    user_ids = []
    with open(user_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Try to convert to int if possible, otherwise keep as string
                try:
                    user_ids.append(int(line))
                except ValueError:
                    user_ids.append(line)

    return user_ids

def main():
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    # Add recommendation-related configs
    if args.mode == 'generate':
        command_line_configs.update({
            'top_k': args.top_k,
            'include_scores': args.include_scores,
            'use_graph_decoding': args.use_graph_decoding,
            'batch_size': args.batch_size,
        })

    if args.mode in ['train', 'evaluate']:
        # Standard training/evaluation pipeline
        pipeline = Pipeline(
            model_name=args.model,
            dataset_name=args.dataset,
            checkpoint_path=args.checkpoint,
            config_dict=command_line_configs
        )

        if args.mode == 'train':
            results = pipeline.run()
        else:  # evaluate mode
            # For evaluate mode, we need to load the model and run evaluation
            results = pipeline.run()

        print(f"\n{args.mode.capitalize()} completed.")
        if 'test_results' in results:
            print("Test results:", results['test_results'])

    elif args.mode == 'generate':
        # Recommendation generation mode
        if not args.checkpoint:
            print("ERROR: Checkpoint path is required for generate mode")
            print("Please specify --checkpoint PATH_TO_MODEL.pth")
            sys.exit(1)

        print(f"Generating recommendations with settings:")
        print(f"  Model: {args.model}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Top-K: {args.top_k}")
        print(f"  Include scores: {args.include_scores}")
        print(f"  Graph decoding: {args.use_graph_decoding}")
        print(f"  Output: {args.output}")

        # Load recommender from checkpoint
        try:
            recommender = Recommender.from_checkpoint(
                checkpoint_path=args.checkpoint,
                model_name=args.model,
                dataset_name=args.dataset,
                config_dict=command_line_configs
            )
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            sys.exit(1)

        # Load user list if provided
        user_subset = None
        if args.user_list:
            try:
                user_subset = load_user_list(args.user_list)
                print(f"  User subset: {len(user_subset)} users from {args.user_list}")
            except Exception as e:
                print(f"ERROR: Failed to load user list: {e}")
                sys.exit(1)

        # Generate recommendations
        try:
            recommendations = recommender.generate_from_test_set(
                top_k=args.top_k,
                include_scores=args.include_scores,
                batch_size=args.batch_size,
                user_subset=user_subset
            )
        except Exception as e:
            print(f"ERROR: Failed to generate recommendations: {e}")
            sys.exit(1)
        # Save recommendations
        try:
            metadata = {
                "command": " ".join(sys.argv),
                "model": args.model,
                "dataset": args.dataset,
                "checkpoint": args.checkpoint,
                "top_k": args.top_k,
                "include_scores": args.include_scores,
                "use_graph_decoding": args.use_graph_decoding,
            }
            recommender.save_recommendations(recommendations, args.output, metadata)
            print(f"\nSuccessfully generated recommendations for {len(recommendations)} users")
            print(f"Output saved to: {args.output}")
        except Exception as e:
            print(f"ERROR: Failed to save recommendations: {e}")
            sys.exit(1)
    else:
        print(f"ERROR: Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == '__main__':
    main()