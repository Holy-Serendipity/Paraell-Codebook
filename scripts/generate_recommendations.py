#!/usr/bin/env python3
"""
Standalone script for generating recommendations from trained RPPG models.

This script provides a simplified interface for generating batch recommendations
and exporting them to JSON format for online testing.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from genrec.recommender import Recommender


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate recommendations from trained RPPG model'
    )

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file (.pth)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file path for recommendations')

    # Model and dataset arguments
    parser.add_argument('--model', type=str, default='RPG',
                       help='Model name (default: RPG)')
    parser.add_argument('--dataset', type=str, default='Netease',
                       help='Dataset name (default: Netease)')

    # Recommendation parameters
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of recommendations per user (default: 10)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for inference (default: 256)')
    parser.add_argument('--include_scores', action='store_true',
                       help='Include confidence scores in output')
    parser.add_argument('--use_graph_decoding', action='store_true',
                       help='Use graph-constrained decoding')

    # User selection
    parser.add_argument('--user_list', type=str, default=None,
                       help='File containing list of user IDs (one per line)')
    parser.add_argument('--all_users', action='store_true',
                       help='Generate recommendations for all test users')

    # Additional configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom configuration YAML file')

    return parser.parse_args()


def load_config_dict(config_file):
    """Load configuration from YAML file."""
    import yaml
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def load_user_list(user_list_file):
    """Load user IDs from file."""
    if not os.path.exists(user_list_file):
        raise FileNotFoundError(f"User list file not found: {user_list_file}")

    user_ids = []
    with open(user_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    user_ids.append(int(line))
                except ValueError:
                    user_ids.append(line)

    return user_ids


def main():
    args = parse_args()

    # Validate arguments
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # Prepare configuration dictionary
    config_dict = {}
    if args.config:
        try:
            config_dict = load_config_dict(args.config)
        except Exception as e:
            print(f"ERROR: Failed to load config file: {e}")
            sys.exit(1)

    # Add command-line arguments to config
    config_dict.update({
        'top_k': args.top_k,
        'include_scores': args.include_scores,
        'use_graph_decoding': args.use_graph_decoding,
        'batch_size': args.batch_size,
    })

    print("=" * 60)
    print("RPPG Recommendation Generator")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Top-K: {args.top_k}")
    print(f"Include scores: {args.include_scores}")
    print(f"Graph decoding: {args.use_graph_decoding}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output}")

    # Load user list if provided
    user_subset = None
    if args.user_list:
        try:
            user_subset = load_user_list(args.user_list)
            print(f"User subset: {len(user_subset)} users from {args.user_list}")
        except Exception as e:
            print(f"ERROR: Failed to load user list: {e}")
            sys.exit(1)
    elif args.all_users:
        print("User subset: All test users")
    else:
        print("User subset: All test users (default)")

    print("-" * 60)

    try:
        # Load recommender
        print("Loading model and dataset...")
        recommender = Recommender.from_checkpoint(
            checkpoint_path=args.checkpoint,
            model_name=args.model,
            dataset_name=args.dataset,
            config_dict=config_dict
        )

        # Generate recommendations
        print("Generating recommendations...")
        recommendations = recommender.generate_from_test_set(
            top_k=args.top_k,
            include_scores=args.include_scores,
            batch_size=args.batch_size,
            user_subset=user_subset
        )

        # Prepare metadata
        metadata = {
            "generator": "scripts/generate_recommendations.py",
            "model": args.model,
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "top_k": args.top_k,
            "include_scores": args.include_scores,
            "use_graph_decoding": args.use_graph_decoding,
            "batch_size": args.batch_size,
        }

        # Save recommendations
        print(f"Saving recommendations to {args.output}...")
        recommender.save_recommendations(recommendations, args.output, metadata)

        # Print summary
        print("\n" + "=" * 60)
        print("SUCCESS: Recommendations generated successfully!")
        print("=" * 60)
        print(f"Total users: {len(recommendations)}")
        print(f"Recommendations per user: {args.top_k}")

        if recommendations and 'recommendations' in recommendations[0]:
            sample_rec = recommendations[0]['recommendations']
            if sample_rec:
                print(f"Sample recommendation (first user):")
                for i, rec in enumerate(sample_rec[:3]):
                    item_id = rec['item_id']
                    score = rec.get('score', 'N/A')
                    print(f"  {i+1}. Item {item_id} (score: {score:.4f})")
                if len(sample_rec) > 3:
                    print(f"  ... and {len(sample_rec) - 3} more")

        print(f"\nOutput file: {args.output}")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()