import os

import pandas as pd
from sklearn.model_selection import ParameterGrid

from feature_restorer_metric_getter import FeatureRestorerMetricGetter

from nb_helper import (display_or_print, load_pickle, mk_dir_if_does_not_exist,
                       save_pickle, try_clear_output)

ERROR_GRID_SEARCH_EXISTS = """\
This NBSpaceRestorer already has a grid search with the name \
{grid_search_name}. Either continue the existing grid search, or choose a \
new grid search name."""
MESSAGE_SKIPPING_PARAMS = """\
Skipping parameter combination at index {i} because results \
are already in the grid search log.
"""

LOG_DF_COLS = ['i', 'L', 'lambda_', 'Precision', 'Recall', 'F-score']


# ====================
class NBSpaceRestorerGridSearch:

    # ====================
    def __init__(self, parent, **kwargs):

        self.parent = parent
        self.__dict__.update(kwargs)
        if os.path.exists(self.root_folder()):
            raise ValueError(
                ERROR_GRID_SEARCH_EXISTS.format(
                    grid_search_name=self.grid_search_name)
            )
        mk_dir_if_does_not_exist(self.root_folder())
        save_pickle(self.ref, self.ref_path())
        save_pickle(self.input, self.input_path())
        self.set_param_combos()
        save_pickle(self.param_combos, self.param_combos_path())
        self.run_grid_search()

    # ====================
    def set_param_combos(self):

        self.param_combos = list(ParameterGrid({
            'L': self.L,
            'lambda_': self.lambda_
        }))

    # ====================
    @classmethod
    def load(cls, parent, grid_search_name: str):

        self = cls.__new__(cls)
        self.parent = parent
        self.grid_search_name = grid_search_name
        self.ref = load_pickle(self.ref_path())
        self.input = load_pickle(self.input_path())
        self.param_combos = load_pickle(self.param_combos_path())
        self.run_grid_search()

    # ====================
    def run_grid_search(self):

        log_df = self.get_log_df()
        for i, parameters in enumerate(self.param_combos):
            try_clear_output()
            display_or_print(log_df)
            if len(log_df[log_df['i'] == i]) > 0:
                print(MESSAGE_SKIPPING_PARAMS.format(i=i))
                continue
            L = parameters['L']
            lambda_ = parameters['lambda_']
            print('L =', L, '; lambda_ =', lambda_)
            hyp = self.parent.restore(self.input, L=L, lambda_=lambda_)
            frmg = FeatureRestorerMetricGetter(
                self.ref, hyp, capitalisation=False, feature_chars=' ',
                get_wer_info_on_init=False
            )
            precision = \
                frmg.get_prfs_all_features()["Spaces (' ')"]['Precision']
            recall = frmg.get_prfs_all_features()["Spaces (' ')"]['Recall']
            f_score = frmg.get_prfs_all_features()["Spaces (' ')"]['F-score']
            log_df = log_df.append({
                'i': i,
                'L': L,
                'lambda_': lambda_,
                'Precision': precision,
                'Recall': recall,
                'F-score': f_score,
            }, ignore_index=True)
            self.save_log(log_df)

    # ====================
    def root_folder(self):

        return os.path.join(
            self.parent.grid_search_path(),
            self.grid_search_name
        )

    # ====================
    def log_path(self):

        return os.path.join(self.root_folder(), 'log.csv')

    # ====================
    def ref_path(self):

        return os.path.join(self.root_folder(), 'ref.pickle')

    # ====================
    def input_path(self):

        return os.path.join(self.root_folder(), 'input.pickle')

    # ====================
    def param_combos_path(self):

        return os.path.join(self.root_folder(), 'param_combos.pickle')

    # ====================
    def get_log_df(self) -> pd.DataFrame:

        log_path = self.log_path()
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
        else:
            log_df = pd.DataFrame(columns=LOG_DF_COLS)
            log_df.to_csv(log_path, index=False)
        return log_df

    # ====================
    def save_log(self, log: pd.DataFrame):

        log.to_csv(self.log_path(), index=False)
