import os


def getDataRow(path_to_dir, line: str) -> str:
    line = line.split()
    return f"{os.path.join(path_to_dir, line[0])}\t{' '.join(line[1:5])} {' '.join(line[15:])}\n"


def writeAnnotations(profile_file_path,
                     image_dir,
                     target_file):
    with open(profile_file_path, 'r') as profile, open(target_file, 'a') as target:
        profiles = profile.readlines()

        for prof in profiles:
            target.write(getDataRow(image_dir, prof))


if __name__ == '__main__':
    profile_train = 'Menpo2D/Train/Menpo2D_profile_train.txt'
    image_dir_train = os.path.abspath('Menpo2D/Train')

    profile_test = 'Menpo2D/Test/Menpo2D_profile_test.txt'
    image_dir_test = os.path.abspath('Menpo2D/Test')

    target = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'annotation_file.txt')

    writeAnnotations(profile_train, image_dir_train, target)
    writeAnnotations(profile_test, image_dir_test, target)
