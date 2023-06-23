import pandas as pd
import seaborn as sns

from pathlib import Path
from methods.io_utils import pickle_load
from methods.results_utils import parse_experiment_name, used_models, explainer_map, model_map, to_latex

sns.set_theme(style="whitegrid")


def main():
    # in this directory the pickle files for experiments should be stored - change if necessary
    experiment_dir = Path(__file__).parent.parent / 'results/experiment6_23/10_perc'
    available_experiments = experiment_dir.glob('./*')
    results = {}
    for result_dir in available_experiments:
        if not (result_dir / 'experiment6_2_k=3.pkl').exists():
            continue
        res = pickle_load(result_dir / 'experiment6_2_k=3.pkl')
        assert res['n_examples'] == 5000, 'failed sanity check: {}'.format(res['n_examples'])

        results[result_dir.name] = res['label_flips']

    formatted_results = {0: {}, 1: {}}
    for model_name in used_models:
        formatted_results[0][model_name] = {}
        formatted_results[1][model_name] = {}

    for exp_name, result in results.items():
        model_name, explainer, _, segmented = parse_experiment_name(exp_name)
        for agg_name, agg_value in result.items():
            if not (explainer == 'lime' or 'magnitude' in agg_name or 'abs' in agg_name):
                continue
            agg_value = agg_value.item()
            formatted_results[int(segmented)][model_name]['{}_{}'.format(explainer, agg_name)] = agg_value

    print('------------------------ rectangular segments ------------------------')
    get_latex_table(0, formatted_results)
    print('------------------------ SLIC segments ------------------------')
    get_latex_table(1, formatted_results)


def get_latex_table(segment, formatted_results):
    """ Creates a latex table that displays the formatted results (for rectangular or SLIC segments). """
    results_df = pd.DataFrame(formatted_results[segment]).T / 5000 * 100
    print('columns', results_df.columns)
    results_df_renamed = results_df.rename(columns=explainer_map, index=model_map)
    to_latex(results_df_renamed[['LIME', 'IG', 'Sal', 'GBP']],
             caption='standard explainers with segm.: {}'.format(segment))
    print('\n')
    to_latex(results_df_renamed[
                 ['IG-SG', 'Sal-SG', 'GBP-SG', 'IG-SG2', 'Sal-SG2', 'GBP-SG2', 'IG-VG', 'Sal-VG', 'GBP-VG']],
             caption='explainers with smoothing with segm.: {}'.format(segment))


if __name__ == '__main__':
    main()
