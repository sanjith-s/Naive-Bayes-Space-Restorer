"""test_nb_space_restorer.py

Smoke tests to check that operations carried out in interactive demo run
on sample data without errors."""

from nb_space_restorer.nb_space_restorer import NBSpaceRestorer
import pandas as pd
from pytest import fixture

train = pd.read_csv('sample_data/train.csv')['reference'].to_list()
test_input = pd.read_csv('sample_data/test.csv')['input'].to_list()
test_ref = pd.read_csv('sample_data/test.csv')['input'].to_list()


# ====================
@fixture(scope='module')
def restorer():

    restorer_ = NBSpaceRestorer(
        train,
        ignore_case=True
    )
    yield restorer_


# ====================
def test_init(restorer):

    restorer


# ====================
def test_grid_search(restorer):

    restorer.add_grid_search(
        'grid_search_1',
        L=[18],
        lambda_=[14.0],
        ref=test_ref,
        input=test_input
    )


# ====================
def test_load_grid_search(restorer):

    restorer.load_grid_search('grid_search_1')


# ====================
def test_show_optimal_params(restorer):

    restorer.show_optimal_params()
