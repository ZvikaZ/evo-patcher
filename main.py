import argparse
from evolution import evolve

if __name__ == "__main__":
    # TODO description and help strings
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                         description='Evolutionary Adversarial Patcher')
    general_group = arg_parser.add_argument_group("General options")
    general_group.add_argument("--random-seed", type=int, default=1)

    evolution_group = arg_parser.add_argument_group("Evolution options")
    evolution_group.add_argument("--num-of-evolve-threads", type=int, default=1)  # TODO optimal value?
    evolution_group.add_argument("--population-size", '-p', type=int, default=200)
    evolution_group.add_argument("--max-generation", '-g', type=int, default=200)
    evolution_group.add_argument("--creation-max-depth", type=int, default=6)

    images_group = arg_parser.add_argument_group("Images options")
    images_group.add_argument("--num-of-images-threads", type=int, default=8)  # TODO increase?
    images_group.add_argument("--imagenet-path", default='/cs_storage/public_datasets/ImageNet')
    images_group.add_argument("--batch-size", type=int, default=40)  # TODO
    images_group.add_argument("--num-of-images", '-n', type=int, default=40)

    yolo_group = arg_parser.add_argument_group("Yolo options")
    images_group.add_argument("--threshold-size-ratio", default=0.1)
    images_group.add_argument("--threshold-confidence", default=0.8)

    args = arg_parser.parse_args()

    evolve(creation_max_depth=args.creation_max_depth,
           population_size=args.population_size,
           num_of_evolve_threads=args.num_of_evolve_threads,
           num_of_images_threads=args.num_of_images_threads,
           max_generation=args.max_generation,
           random_seed=args.random_seed,
           imagenet_path=args.imagenet_path,
           batch_size=args.batch_size,
           num_of_images=args.num_of_images,
           threshold_size_ratio=args.threshold_size_ratio,
           threshold_confidence=args.threshold_confidence)
