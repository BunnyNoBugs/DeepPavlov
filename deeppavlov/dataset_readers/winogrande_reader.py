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

from typing import Optional
from logging import getLogger
from pathlib import Path

import pandas as pd
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download

log = getLogger(__name__)


class WinograndeReader(DatasetReader):
    """
    Dataset reader for Winogrande
    """

    @overrides
    def read(self, data_path: str, format: str = 'jsonl',
             train: str = 'train_m',
             valid: Optional[str] = None,
             test: Optional[str] = None,
             *args, **kwargs) -> dict:
        """
        Read dataset
        """
        num_options = 2

        split_names = {'train': train, 'valid': valid, 'test': test}

        data = {'train': None,
                'valid': None,
                'test': None}
        for split in split_names:
            split_name = split_names[split]
            if split_name:
                file_name = f'{split_name}.{format}'
                file = Path(data_path, file_name)
                if file.exists():
                    if format == 'jsonl':
                        df = pd.read_json(file, lines=True)
                        sentences = [[sentence] * num_options for sentence in df['sentence']]
                        options = [[option1, option2] for option1, option2 in zip(df['option1'], df['option2'])]
                        if 'answer' in df:
                            answers = df['answer'].tolist()
                        else:
                            answers = [-1] * len(df)

                        data[split] = [{'sentences': s, 'options': o, 'answer': a} for s, o, a
                                       in zip(sentences, options, answers)]
                    else:
                        raise NotImplementedError(f'{format} not supported')

        return data
