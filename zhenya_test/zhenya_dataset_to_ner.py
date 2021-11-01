import pandas as pd
from nltk.tokenize.treebank import TreebankWordTokenizer
import json
from textdistance import levenshtein


def convert_to_ner(load_path: str, save_path: str):
    tokenizer = TreebankWordTokenizer()

    df = pd.read_json(load_path, encoding='utf-8', lines=True)
    df = df[df['dialog_acts'].apply(lambda x: x[0]['act'] != 'api_call')]

    ner_output = ''

    with open('../zhenya_dataset/dstc_ru_slots.json') as f:
        all_slots = json.load(f)
    slot_subs = []
    for i in all_slots:
        slot_subs.extend(list(all_slots[i].keys()))

    for _, row in df.iterrows():
        text = row['text']
        slots = row['dialog_acts'][0]['slots']
        slot_spans = []
        if slots:
            for label, span in slots:
                span_start = text.find(span)
                if span_start == -1:
                    for span_sub in sorted(slot_subs, key=lambda x: levenshtein(x, span))[1:]:
                        span_start = text.find(span_sub)
                        if span_start != -1:
                            break
                    span_start = text.find(span_sub)
                    span_stop = span_start + len(span_sub)
                else:
                    span_stop = span_start + len(span)
                slot_spans.append([label, (span_start, span_stop)])

        token_spans = tokenizer.span_tokenize(text)
        for token_span in token_spans:
            token = text[token_span[0]:token_span[1]]
            ner_output += token
            ner_output += ' '
            for slot_span in slot_spans:
                if slot_span[1][0] <= token_span[0] < slot_span[1][1]:
                    ner_output += slot_span[0]
                    break
            else:
                ner_output += 'O'
            ner_output += '\n'
        ner_output += '\n'

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(ner_output)


def main():
    convert_to_ner('../zhenya_dataset/dstc2-trn.jsonlist', '../zhenya_dataset/ner/train.txt')
    convert_to_ner('../zhenya_dataset/dstc2-val.jsonlist', '../zhenya_dataset/ner/valid.txt')
    convert_to_ner('../zhenya_dataset/dstc2-tst.jsonlist', '../zhenya_dataset/ner/test.txt')


if __name__ == '__main__':
    main()
