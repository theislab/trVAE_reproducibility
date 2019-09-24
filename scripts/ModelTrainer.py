import subprocess
import sys

TRAINING_DICT = {
    "trVAE": ["haber", "kang"],
    "scGen": ["haber", "kang"],
    "cvae": ["haber", "kang"],
    "cyclegan": ["haber", "kang"],
    "scVI": ["haber", "kang"],
    "mmdcvae": ["haber", "kang"],
    "saucie": ["haber", "kang"],
    "DCtrVAE": ["mnist", "celeba"],
}


def train(method, dataset):
    if TRAINING_DICT.keys().__contains__(method):
        if TRAINING_DICT[method].__contains__(dataset):
            command = f"python -m scripts.train_{method} {dataset}"
            subprocess.call([command], shell=True)


def main():
    if len(sys.argv) < 2:
        model_to_train = "all"
        dataset_to_train = None
    else:
        model_to_train = sys.argv[1]
        dataset_to_train = sys.argv[2]
    if model_to_train == "all":
        train('trVAE', 'haber')
        train('trVAE', 'kang')

        train('cyclegan', 'haber')
        train('cyclegan', 'kang')

        train('DCtrVAE', 'haber')
        train('DCtrVAE', 'kang')

        train('mmdcvae', 'haber')
        train('mmdcvae', 'kang')

        train('saucie', 'haber')
        train('saucie', 'kang')

        train('scGen', 'haber')
        train('scGen', 'kang')

        train('scVI', 'haber')
        train('scVI', 'kang')

    else:
        train(model_to_train, dataset_to_train)


if __name__ == '__main__':
    main()
