import os


def main():
    data_path = ''
    adv_path = ''
    archs = ['alexnet', 'vgg', 'resnet', 'densenet', 'inceptionv3']
    explainers = ['lime', 'saliency', 'ig', 'gbp']
    segmentations = ['--segment', '']

    for arch in archs:
        for explainer in explainers:
            for segmentation in segmentations:
                command = 'python -m experiments.label_flip_exps --data-path {} --adv-path {} ' \
                          '--arch {} --explainer {} {}'.format(data_path, adv_path, arch, explainer, segmentation)
                if explainer == 'lime':
                    command += ' --baseline'
                print(command)
                os.system(command)


if __name__ == '__main__':
    main()

