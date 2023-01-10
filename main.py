import argparse

from attack import attack
from misc import set_logger

logger = set_logger(__file__)

if __name__ == "__main__":
    # TODO description and help strings
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                         description='Evolutionary Adversarial Patcher')
    general_group = arg_parser.add_argument_group("General options")
    general_group.add_argument("--random-seed", type=int, default=1)

    evolution_group = arg_parser.add_argument_group("Evolution options")
    evolution_group.add_argument("--patch-ratio-x", type=float, default=0.4)
    evolution_group.add_argument("--patch-ratio-y", type=float, default=0.4)
    evolution_group.add_argument("--elitism-rate", type=float, default=0)
    evolution_group.add_argument("--num-of-evolve-threads", type=int, default=1)  # TODO 2?
    evolution_group.add_argument("--population-size", '-p', type=int, default=500)
    evolution_group.add_argument("--max-generation", '-g', type=int, default=250)
    evolution_group.add_argument("--creation-max-depth", type=int, default=4)
    evolution_group.add_argument("--bloat-weight", '-w', type=float, default=0.0001)  # TODO

    images_group = arg_parser.add_argument_group("Images reading options")
    images_group.add_argument("--num-of-images-threads", type=int, default=4)  # TODO increase?
    images_group.add_argument("--imagenet-path", default='/cs_storage/public_datasets/ImageNet')
    images_group.add_argument("--batch-size", type=int, default=100)  # 500 is too big to always fit in memory
    images_group.add_argument("--num-of-images", '-n', type=int, default=40)
    images_group.add_argument("--classes", nargs='*',
                              default=['freight car', 'passenger car', 'sports car', 'streetcar', ])
    images_group.add_argument("--single-image", '-1', action='store_true',
                              help="Perform a single image evolution (instead of Universal); ignores '--classes'")

    yolo_group = arg_parser.add_argument_group("Yolo options")
    images_group.add_argument("--threshold-size-ratio", type=float, default=0.1)
    images_group.add_argument("--threshold-confidence", type=float, default=0.8)

    args = arg_parser.parse_args()
    logger.debug(args)

    attack(single_image=args.single_image,
           creation_max_depth=args.creation_max_depth,
           population_size=args.population_size,
           num_of_evolve_threads=args.num_of_evolve_threads,
           num_of_images_threads=args.num_of_images_threads,
           max_generation=args.max_generation,
           random_seed=args.random_seed,
           patch_ratio_x=args.patch_ratio_x,
           patch_ratio_y=args.patch_ratio_y,
           elitism_rate=args.elitism_rate,
           bloat_weight=args.bloat_weight,
           imagenet_path=args.imagenet_path,
           batch_size=args.batch_size,
           num_of_images=args.num_of_images,
           classes=args.classes,
           threshold_size_ratio=args.threshold_size_ratio,
           threshold_confidence=args.threshold_confidence)
