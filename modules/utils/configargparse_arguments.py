import configargparse

def build_configargparser(parser):
    module_group = parser.add_argument_group(title='Module options')
    model_group = parser.add_argument_group(title='Model options')
    dataset_group = parser.add_argument_group(title='Dataset options')
    trainer_group = parser.add_argument_group(title="'Trainer options")

    module_group.add_argument("--module", type=str, required=True)

    model_group.add_argument("--model", type=str, required=True)

    dataset_group.add_argument("--dataset", type=str, required=True)
    dataset_group.add_argument("--data_root", type=str, default="/media/student/Videodaten/data/cholec80_preprocessed/")
    dataset_group.add_argument("--num_workers", default=0, type=int)

    trainer_group.add_argument("--output_path", type=str, default="logs")
    trainer_group.add_argument("--max_epochs", default=1000, type=int)
    trainer_group.add_argument("--min_epochs", default=1, type=int)
    trainer_group.add_argument("--subproject_name", type=str, default="no_subproject")
    trainer_group.add_argument("--use_deterministic_torch_algorithms", action="store_true")

    known_args, _ = parser.parse_known_args()
    return parser, known_args