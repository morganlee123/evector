from encoder.preprocess import preprocess_librispeech, preprocess_voxceleb1, preprocess_voxceleb2, preprocess_msppod
from utils.argutils import print_args
from pathlib import Path
import argparse


if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms and "
                    "writes them to the disk. This will allow you to train the encoder. The "
                    "datasets required are at least one of VoxCeleb1, VoxCeleb2, MSP-Podcast, and LibriSpeech. "
                    "You should extract them as they are "
                    "after having downloaded them and put them in a same directory, e.g.:\n"
                    "-[datasets_root]\n"
                    "  -LibriSpeech\n"
                    "    -train-other-500\n"
                    "  -VoxCeleb1\n"
                    "    -wav\n"
                    "    -vox1_meta.csv\n"
                    "  -VoxCeleb2\n"
                    "    -dev"
                    "  -MSPPod\n"
                    "    -Audios",
        formatter_class=MyFormatter
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your datasets.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms. If left out, "
        "defaults to <datasets_root>/EVec/encoder/")
    parser.add_argument("-d", "--datasets", type=str,
                        default="librispeech_other,voxceleb1,voxceleb2,msppod", help=\
        "Comma-separated list of the name of the datasets you want to preprocess. Only the train "
        "set of these datasets will be used. Possible names: librispeech_other, voxceleb1, "
        "voxceleb2, msppod.")
    #parser.add_argument("-cs", "--speaker_list", type=str,
    #                    default="0", help=\
    #    "Choose a particular speaker to train the model upon. This can potentially improve the performance for the aforementioned speaker")

    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to skip existing output files with the same name. Useful if this script was "
        "interrupted.")
    print('Parsing Arguments!')
    args = parser.parse_args()
    print('Arguments Parsed!')

    # NOTE: This is what you run in the terminal to preprocess the MSP-Podcast dataset. This preprocessing consumes 
    # ~50 GB worth of .npy files with the current size of the dataset as of 1/31/2023
    # python encoder_preprocess.py "/research/iprobe/datastore/datasets/speech/" -d msppod -o "/research/iprobe-sandle20/Playground/evector/Data/EVec/encoder/"


    # Process the arguments
    args.datasets = args.datasets.split(",")
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("EVec", "encoder") # default path
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Preprocess the datasets
    print_args(args, parser)
    preprocess_func = {
        "msppod": preprocess_msppod,
        "librispeech_other": preprocess_librispeech,
        "voxceleb1": preprocess_voxceleb1,
        "voxceleb2": preprocess_voxceleb2,
    }
    args = vars(args)
    for dataset in args.pop("datasets"):
        print("Preprocessing %s" % dataset)
        preprocess_func[dataset](**args)
