# Naive Bayes Space Restorer

A Python library for training Naive Bayes-based statistical machine learning models for restoring spaces to unsegmented sequences of input characters.

E.g.
`thisisasentence -> this is a sentence`

Developed and used for the paper "Comparison of Token- and Character-Level Approaches to Restoration of Spaces, Punctuation, and Capitalization in Various Languages", which is scheduled for publication in December 2022.

The model is based on the description and Python code in Norvig (2009), and the chunking methods for handling long strings is borrowed from Jenks (2018).

The implementation here allows for easy restoration of spaces to entire datasets of documents with a progress bar, and for tuning of hyperparameters _L_ (maximum word length) and λ (smoothing parameter) for model optimization.

## Interactive demo

The quickest and best way to get acquainted with the library is through the interactive demo [here](https://colab.research.google.com/drive/1ngcioFhOvS95oSYjkC4kqIYtBZUFygx6?usp=sharing), where you can walk through the steps involved in using the library and clean some sample data from the Ted Talks dataset used in the paper.

Alternatively, scroll down for instructions on getting started and basic documentation.

## Getting started

### Install the library using `pip`

```
!pip install git+https://github.com/ljdyer/Naive-Bayes-Space-Restorer.git
```

### Import the `NBSpaceRestorer` class

```python
from nb_space_restorer import NBSpaceRestorer
```

## Model training, optimization, and inference using the `NBSpaceRestorer` class

### Initialize and train a model

#### `NBSpaceRestorer.__init__`

```python
    # ====================
    def __init__(self,
                 train_texts: list,
                 ignore_case: bool = True,
                 save_path: Optional[str] = None):
        """Initalize and train an instance of the class.

        Args:
          train_texts (list):
            The list of 'gold standard' documents (running text with spaces)
            on which to train the model.
          ignore_case (bool, optional):
            Whether or not to ignore case during training (so that e.g.
            'banana', 'Banana', and 'BANANA' are all counted as instances
            of 'banana'). Defaults to True.
          save_path (Optional[str], optional):
            The path to a pickle file to save the model to. Defaults to None.
        """
```

#### Example usage:

```python
restorer = NBSpaceRestorer(
    train_texts=train['reference'].to_list(),
    ignore_case=True
)
```

<img src="readme-img/01-init.PNG"></img>

### Run a grid search to find optimal hyperparameters for inference

#### `NBSpaceRestorer.add_grid_search`

```python
    # ====================
    def add_grid_search(self,
                        grid_search_name: str,
                        L: List[int],
                        lambda_: List[float],
                        ref: List[str],
                        input: List[str]):

        self.grid_searches[grid_search_name] = {}
        self.current_grid_search_name = grid_search_name
        self.current_grid_search()['param_values'] = {
            'L': L,
            'lambda': lambda_
        }
        param_combos = list(ParameterGrid({
            'L': L,
            'lambda': lambda_
        }))
        self.current_grid_search()['param_combos'] = \
            {i: pc for i, pc in enumerate(param_combos)}
        self.current_grid_search()['results'] = \
            {i: None for i in range(len(param_combos))}
        self.save()
        self.run_grid_search(ref, input)

    # ====================
    def load_grid_search(self, grid_search_name: str):

        self.grid_search_name = grid_search_name
        completed, total = self.param_combos_completed()
        if total > completed:
            print(MESSAGE_GRID_SEARCH_INCOMPLETE.format(
                grid_search_name=grid_search_name,
                num_untested=total-completed
            ))

    # ====================
    def current_grid_search(self):

        return self.grid_searches[self.current_grid_search_name]

    # ====================
    def run_grid_search(self, ref: List[str], input: List[str]):

        param_combos = self.current_grid_search()['param_combos']
        L_start = self.L
        lambda_start = self.lambda_
        for i, parameters in param_combos.items():
            self.running_grid_search = True
            try_clear_output()
            display_or_print(self.grid_search_results_df())
            self.show_param_combos_completed()
            print()
            if self.current_grid_search()['results'][i] is not None:
                print(MESSAGE_SKIPPING_PARAMS.format(i=i))
                self.running_grid_search = False
                continue
            L = parameters['L']
            lambda_ = parameters['lambda']
            print('L =', L, '; lambda =', lambda_)
            start_time = time.time()
            hyp = self.restore(input, L=L, lambda_=lambda_)
            evaluator = FeatureRestorationEvaluator(
                ref,
                hyp,
                capitalization=False,
                feature_chars=' ',
                get_wer_info_on_init=False
            )
            prf = evaluator.get_prfs()[' ']
            time_taken = time.time() - start_time
            self.current_grid_search()['results'][i] = {
                'i': i, 'L': L, 'lambda': lambda_,
                **prf, 'Time (s)': time_taken
            }
            self.restore_chunk.cache_clear()
            self.set_L(L_start)
            self.set_lambda(lambda_start)
            self.running_grid_search = False
            self.save()
        try_clear_output()
        display_or_print(self.grid_search_results_df())
        self.show_param_combos_completed()

    # ====================
    def grid_search_results_df(self) -> pd.DataFrame:
        """Get the results of the current grid search.

        Returns:
          pd.DataFrame
            A pandas dataframe containing the results for all the
            parameter combinations tested so far.
        """
```

#### Example usage:

```python
restorer.add_grid_search(
    grid_search_name='grid_search_1',
    L=[18, 20, 22],
    lambda_=[8.0, 10.0, 12.0],
    ref=test_ref,
    input=test_input
)
```

<img src="readme-img/02-add_grid_search.PNG"></img>

### Show optimal hyperparameters from the current grid search

#### `NBSpaceRestorer.show_optimal_params

```python
    # ====================
    def show_optimal_params(self,
                            metric_to_optimize: Optional[str] = None,
                            min_or_max: Optional[str] = None):
        """Display the rows from the grid search results table with the best
        results based on the values of the metric_to_optimize and min_or_max
        attributes of the class instance, and the values of the hyperparameters
        that produce those results.
        If there is more than one hyperparameter combination that produces the
        best result for metric_to_optimize, the one that was tested first will
        be selected.

        Args:
          metric_to_optimize (Optional[str], optional):
            If provided, the metric_to_optimize attribute of the class
            instance will be set to this value before finding the optimal
            hyperparameter values.
            Defaults to None.
          min_or_max (Optional[str], optional):
            If provided, the min_or_max attribute of the class
            instance will be set to this value before finding the optimal
            hyperparameter values. Defaults to None.
        """
```

#### Example usage:

```python
restorer.show_optimal_params(metric_to_optimize='Recall')
```

<img src="readme-img/03-show_optimal_params.PNG"></img>

### Apply the optimal hyperparameters from the current grid search

#### `NBSpaceRestorer.set_optimal_params`

```python
    # ====================
    def set_optimal_params(self,
                           metric_to_optimize: Optional[str] = None,
                           min_or_max: Optional[str] = None):
        """Set the L and lambda_ attributes of the class instances to the
        optimal hyperparameters for the model based on the values of the
        metric_to_optimize and min_or_max attributes of the class instance.

        If there is more than one hyperparameter combination that produces the
        best result for metric_to_optimize, the one that was tested first will
        be selected.

        Args:
          metric_to_optimize (Optional[str], optional):
            If provided, the metric_to_optimize attribute of the class
            instance will be set to this value before finding the optimal
            hyperparameter values.
            Defaults to None.
          min_or_max (Optional[str], optional):
            If provided, the min_or_max attribute of the class
            instance will be set to this value before finding the optimal
            hyperparameter values. Defaults to None.
        """
```

#### Example usage:

```python
restorer.set_optimal_params()
```

<img src="readme-img/04-set_optimal_params.PNG"></img>

### Load a previously saved model from a pickle file

#### `NBSpaceRestorer.load`

```python
    # ====================
    def load_grid_search(self, grid_search_name: str):

        self.grid_search_name = grid_search_name
        completed, total = self.param_combos_completed()
        if total > completed:
            print(MESSAGE_GRID_SEARCH_INCOMPLETE.format(
                grid_search_name=grid_search_name,
                num_untested=total-completed
            ))

    # ====================
    def current_grid_search(self):

        return self.grid_searches[self.current_grid_search_name]

    # ====================
    def run_grid_search(self, ref: List[str], input: List[str]):

        param_combos = self.current_grid_search()['param_combos']
        L_start = self.L
        lambda_start = self.lambda_
        for i, parameters in param_combos.items():
            self.running_grid_search = True
            try_clear_output()
            display_or_print(self.grid_search_results_df())
            self.show_param_combos_completed()
            print()
            if self.current_grid_search()['results'][i] is not None:
                print(MESSAGE_SKIPPING_PARAMS.format(i=i))
                self.running_grid_search = False
                continue
            L = parameters['L']
            lambda_ = parameters['lambda']
            print('L =', L, '; lambda =', lambda_)
            start_time = time.time()
            hyp = self.restore(input, L=L, lambda_=lambda_)
            evaluator = FeatureRestorationEvaluator(
                ref,
                hyp,
                capitalization=False,
                feature_chars=' ',
                get_wer_info_on_init=False
            )
            prf = evaluator.get_prfs()[' ']
            time_taken = time.time() - start_time
            self.current_grid_search()['results'][i] = {
                'i': i, 'L': L, 'lambda': lambda_,
                **prf, 'Time (s)': time_taken
            }
            self.restore_chunk.cache_clear()
            self.set_L(L_start)
            self.set_lambda(lambda_start)
            self.running_grid_search = False
            self.save()
        try_clear_output()
        display_or_print(self.grid_search_results_df())
        self.show_param_combos_completed()

    # ====================
    def grid_search_results_df(self) -> pd.DataFrame:
        """Get the results of the current grid search.

        Returns:
          pd.DataFrame
            A pandas dataframe containing the results for all the
            parameter combinations tested so far.
        """
```

#### Example usage:

```python
NB_TedTalks = NBSpaceRestorer.load(
    'https://raw.githubusercontent.com/ljdyer/Naive-Bayes-Space-Restorer/main/NB_TedTalks.pickle',
    read_only=True
)
```

<img src="readme-img/05-load.PNG"></img>

### Restore spaces to an unsegmented sequence of input characters

#### `NBSpaceRestorer.restore`

```python
    # ====================
    def restore(self,
                texts: Union[str, List[str]],
                L: Optional[int] = None,
                lambda_: Optional[int] = None) -> Union[str, List[str]]:
        """Restore spaces to either a single string, or a list of
        strings.

        Args:
          texts (Union[str, List[str]]):
            Either a single string of input characters not containing spaces
            (e.g. 'thisisasentence') or a list of such strings
          L (Optional[int], optional):
            The value of the hyperparameter L to set before restoring
          lambda_ (Optional[float], optional):
            The value of the hyperparameter lambda_ to set before restoring

        Returns:
          Union[str, List[str]]:
            The string or list of strings with spaces restored
        """
```

#### Example usage:

```python
NB_TedTalks.restore(test_input)
```

<img src="readme-img/06-restore.PNG"></img>

## References

G. Jenks, ”python-wordsegment,” July, 2018. [Online]. Available:
https://github.com/grantjenks/python-wordsegment. [Accessed May
2, 2022].

P. Norvig, “Natural language corpus data,” in Beautiful Data, T.
Segaran and J. Hammerbacher, Eds. Sebastopol: O’Reilly, 2009, pp.
219-242.
