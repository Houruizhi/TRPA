import argparse

def get_args_from_parser():
    parser = argparse.ArgumentParser(description="Train the denoising model")
    parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            nargs='+',
            default=[],
        )
    parser.add_argument(
            "--config_file",
            help="The path of .yaml config file",
            default='',
        )
    args = parser.parse_args()
    return args
