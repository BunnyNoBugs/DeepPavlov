import deeppavlov
from deeppavlov import train_model

model = train_model(deeppavlov.configs.squad.refactor_squad_torch_bert)
