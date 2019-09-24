import os

import wget

url_dict = {
    "haber_normalized": "https://hmgubox.helmholtz-muenchen.de/lib/96bd2d06-9efd-4731-8776-4d4e5b1ee3d3/file/haber_normalized.h5ad?dl=1",
    "haber_count": "https://hmgubox.helmholtz-muenchen.de/lib/96bd2d06-9efd-4731-8776-4d4e5b1ee3d3/file/haber_count.h5ad?dl=1",
    "kang_normalized": "https://hmgubox.helmholtz-muenchen.de/lib/96bd2d06-9efd-4731-8776-4d4e5b1ee3d3/file/kang_normalized.h5ad?dl=1",
    "kang_count": "https://hmgubox.helmholtz-muenchen.de/lib/96bd2d06-9efd-4731-8776-4d4e5b1ee3d3/file/kang_count.h5ad?dl=1",
    "celeba": "https://hmgubox.helmholtz-muenchen.de/lib/96bd2d06-9efd-4731-8776-4d4e5b1ee3d3/file/celeba_Smiling_32x32_50000.h5ad?dl=1",
    "mnist": "https://hmgubox.helmholtz-muenchen.de/lib/96bd2d06-9efd-4731-8776-4d4e5b1ee3d3/file/thick_thin_mnist.h5ad?dl=1",
}


def download_data(data_name, key=None):
    data_path = "../data/"
    if key is None:
        data_path = os.path.join(data_path, f"{data_name}.h5ad")
        data_url = url_dict[f"{data_name}"]

        if not os.path.exists(data_path):
            wget.download(data_url, data_path)
    else:
        data_path = os.path.join(data_path, f"{key}.h5ad")
        data_url = url_dict[key]

        if not os.path.exists(data_path):
            wget.download(data_url, data_path)
    print(f"{data_name} data has been downloaded and saved in {data_path}")


def main():
    data_names = ["haber_normalized", 'haber_count', 'kang_normalized', 'kang_count', 'celeba', 'mnist']
    for data_name in data_names:
        download_data(data_name)


if __name__ == '__main__':
    main()
