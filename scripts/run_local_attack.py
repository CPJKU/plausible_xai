import os


def main():
    data_path = ''
    adv_path = ''
    explainers = ['lime', 'saliency', 'ig', 'gbp', 'saliency_sm', 'saliency_smsq', 'saliency_var',
                  'ig_sm', 'ig_smsq', 'ig_var', 'gbp_sm', 'gbp_smsq', 'gbp_var']

    for ex in explainers:
        command = 'python -m experiments.localised_attack --data-path {} --adv-path {} ' \
                  '--arch inceptionv3 --explainer {} --aggregation mean_abs'.format(data_path, adv_path, ex)
        print(command)
        os.system(command)


if __name__ == '__main__':
    main()
