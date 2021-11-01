import deeppavlov
from deeppavlov import train_model

model = train_model(deeppavlov.configs.ner.zhenya_ner)
