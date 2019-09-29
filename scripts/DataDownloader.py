import os

import wget

url_dict = {
    "haber_normalized": "https://drive.google.com/file/d/13d7B1yeHAlI1_AquBif1MRetzJx-hzyp/view?usp=sharing",
    "haber_count": "https://drive.google.com/file/d/1lh_6oRcuBR2KUzDSkvcnZmrFHTMs_2eJ/view?usp=sharing",
    "kang_normalized": "https://drive.google.com/file/d/1ZgzHQU5ubZdQT2o8psWTP0gh5PoJZlCD/view?usp=sharing",
    "kang_count": "https://drive.google.com/file/d/1bd8R8hy8GL3QPgTVdHWtFqdPqbGUdLE9/view?usp=sharing",
    "celeba": "https://drive.google.com/uc?export=download&confirm=mQTe&id=1TU46VNpIRTeYjkisvQyDMf_fCuH1EjQT",
    "mnist": "https://drive.google.com/file/d/1U_AxtWnPn4pAS51-fHmpetQZeHbGJ2xN/view?usp=sharing",
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
