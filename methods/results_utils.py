used_models = ['alexnet', 'vgg', 'resnet', 'densenet', 'inceptionv3']

explainer_map = {
    'lime_mean': 'LIME',
    'lime-eucl_mean': 'LIME (Eucl.)',
    'ig_sm_mean_abs': 'IG-SG',
    'saliency_sm_mean_abs': 'Sal-SG',
    'gbp_sm_mean_abs': 'GBP-SG',
    'gbp_smsq_mean_abs': 'GBP-SG2',
    'saliency_smsq_mean_abs': 'Sal-SG2',
    'ig_smsq_mean_abs': 'IG-SG2',
    'gbp_var_mean_abs': 'GBP-VG',
    'saliency_var_mean_abs': 'Sal-VG',
    'ig_var_mean_abs': 'IG-VG',
    'ig_mean_abs': 'IG',
    'saliency_mean_abs': 'Sal',
    'gbp_mean_abs': 'GBP',
    'magnitude': 'Magnitude (baseline)',
    'magnitude_mean': 'Magnitude (baseline)'
}

model_map = {
    'alexnet': 'AlexNet',
    'vgg': 'VGG16',
    'resnet': 'ResNet-50',
    'densenet': 'DenseNet161',
    'inceptionv3': 'Inception v3'
}

color_map = {
    'lime': 'tab:blue',
    'lime-eucl': 'tab:blue',
    'magnitude': 'tab:orange',
    'ig': 'tab:red',
    'gbp': 'tab:green',
    'saliency': 'tab:purple'
}

def smoothing_symbol_map(smooth):
    if smooth == 'var':
        return 'o'
    elif smooth == 'sm':
        return 'x'
    elif smooth == 'smsq':
        return 's'
    return 'v'

def smoothing_linestyle_map(smooth):
    if smooth == 'var':
        return 'dotted'
    elif smooth == 'sm':
        return 'dashed'
    elif smooth == 'smsq':
        return 'dashdot'
    return 'solid'

def parse_experiment_name(experiment_dir):
    splitted = experiment_dir.split('_')
    segmented = 'segment' in experiment_dir
    model_name = splitted[0]
    if segmented:
        explainer, sub = splitted[1:-2], splitted[-2]
    else:
        explainer, sub = splitted[1:-1], splitted[-1]
    explainer = '_'.join(explainer)
    sub = int(sub.replace('sub', ''))
    return model_name, explainer, sub, segmented


def get_result_name_from_file_name(explainer, file_name, segmented):
    if 'magnitude_mean' in file_name:
        return 'magnitude_mean'
    if 'magnitude' in file_name:
        return 'magnitude'
    aggregation = 'mean'
    if 'pos' in file_name:
        aggregation = aggregation + '_' + 'pos'
    if 'abs' in file_name:
        aggregation = aggregation + '_' + 'abs'
    return '{}_{}'.format(explainer, aggregation, int(segmented))


def to_latex(df, caption=None):
    latex = df.to_latex()
    latex = latex.replace('toprule', 'hline').replace('midrule', 'hline').replace('bottomrule', 'hline')
    latex = latex.replace('{}', 'Model')

    latex = '\\begin{table} \centering \n' + latex + '\end{table}'
    latex = latex.replace('SG2', 'SG$^2$')
    print(latex)
