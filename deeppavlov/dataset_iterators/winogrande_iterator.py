# Copyright 2020 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Any, Union

from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


class WinograndeIterator(DataLearningIterator):
    """Dataset iterator for Winogrande"""

    def preprocess(self, data: List[Tuple[Any, Any]],
                   features: Union[str, List[str]] = ['sentences', 'options'],
                   label: str = 'answer',
                   use_label_name: bool = True,
                   *args, **kwargs) -> List[Tuple[Any, Any]]:
        """Extracts features and labels from Winogrande dataset"""
        dataset = []
        for example in data:
            if isinstance(features, str):
                feat = example[features]
            elif isinstance(features, list):
                feat = tuple(example[f] for f in features)
            else:
                raise RuntimeError(f"features should be str or list, but found: {features}")
            lb = example[label]
            if use_label_name and lb != -1:
                # -1 label is used if there is no label (test set)
                lb = 'option' + str(lb)
            dataset += [(feat, lb)]
        return dataset
