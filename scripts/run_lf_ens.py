import os


def main():
    data_path = ''
    adv_path = ''
    archs = ['alexnet', 'vgg', 'resnet', 'densenet']
    explainers = ['saliency_sm', 'saliency_smsq', 'saliency_var', 'ig_sm',
                  'ig_smsq', 'ig_var', 'gbp_sm', 'gbp_smsq', 'gbp_var']
    segmentations = ['--segment', '']

    for arch in archs:
        for explainer in explainers:
            for segmentation in segmentations:
                command = 'python -m experiments.label_flip_exps --data-path {} --adv-path {} ' \
                          '--arch {} --explainer {} {}'.format(data_path, adv_path, arch, explainer, segmentation)
                print(command)
                os.system(command)


if __name__ == '__main__':
    main()

