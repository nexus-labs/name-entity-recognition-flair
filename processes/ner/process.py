from typing import Dict, List, OrderedDict

from flair.data import Sentence
from flair.models import SequenceTagger
from podder_task_foundation import Context, Payload
from podder_task_foundation import Process as ProcessBase


class Process(ProcessBase):
    def initialize(self, context: Context) -> None:
        self.model = SequenceTagger.load(
            context.config.get("parameters.model_path"))

    def tagExtract(
            self, tagged_documents: List[OrderedDict[str,
                                                     str]]) -> Dict[str, str]:
        res_pred = {}
        for i in range(len(tagged_documents[0]['words'])):
            for ind, val in enumerate(tagged_documents[0]['words'][i]['tags']):
                if val != 'O':
                    if val[2:] not in res_pred:
                        res_pred[val[2:]] = tagged_documents[0]['words'][i][
                            'value'].split()[ind]
                    else:
                        res_pred[val[2:]] = res_pred[val[2:]] + ' ' + \
                                            tagged_documents[0]['words'][i]['value'].split()[ind]

        return res_pred

    def postprocess(self, document: OrderedDict[str, str],
                    res_pred: Dict[str, str]) -> Dict[str, str]:
        if 'SURNAME' in res_pred:
            if 'GIVEN_NAME' in res_pred:
                a = [res_pred['SURNAME'], 0]
                b = [res_pred['GIVEN_NAME'], 0]
                for i in document["words"]:
                    if a[1] != 0 and b[1] != 0:
                        if a[1] > b[1]:
                            res_pred['SURNAME'] = b[0]
                            res_pred['GIVEN_NAME'] = a[0]
                        break
                    elif i['value'] == res_pred['SURNAME']:
                        a[1] = i['bbox']['left'] + i['bbox']['top']
                    elif i['value'] == res_pred['GIVEN_NAME']:
                        b[1] = i['bbox']['left'] + i['bbox']['top']
            elif len(res_pred['SURNAME'].split()) > 1:
                a = [res_pred['SURNAME'].split()[0], 0]
                b = [res_pred['SURNAME'].split()[1], 0]
                for i in document["words"]:
                    if a[1] != 0 and b[1] != 0:
                        if a[1] > b[1]:
                            res_pred['SURNAME'] = b[0]
                            res_pred['GIVEN_NAME'] = a[0]
                        else:
                            res_pred['SURNAME'] = a[0]
                            res_pred['GIVEN_NAME'] = b[0]
                        break
                    if len(i['value'].split()) > 1:
                        for j in i['value'].split():
                            if j == res_pred['SURNAME'].split()[0]:
                                a[1] = i['bbox']['left'] + i['bbox']['top']
                            elif j == res_pred['SURNAME'].split()[1]:
                                b[1] = i['bbox']['left'] + i['bbox']['top']
                    else:
                        if i['value'] == res_pred['SURNAME'].split()[0]:
                            a[1] = i['bbox']['left'] + i['bbox']['top']
                        elif i['value'] == res_pred['SURNAME'].split()[1]:
                            b[1] = i['bbox']['left'] + i['bbox']['top']
        elif 'GIVEN_NAME' in res_pred:
            if len(res_pred['GIVEN_NAME'].split()) > 1:
                a = [res_pred['GIVEN_NAME'].split()[0], 0]
                b = [res_pred['GIVEN_NAME'].split()[1], 0]
                for i in document["words"]:
                    if a[1] != 0 and b[1] != 0:
                        if a[1] > b[1]:
                            res_pred['SURNAME'] = b[0]
                            res_pred['GIVEN_NAME'] = a[0]
                        else:
                            res_pred['SURNAME'] = a[0]
                            res_pred['GIVEN_NAME'] = b[0]
                        break
                    if len(i['value'].split()) > 1:
                        for j in i['value'].split():
                            if j == res_pred['GIVEN_NAME'].split()[0]:
                                a[1] = i['bbox']['left'] + i['bbox']['top']
                            elif j == res_pred['GIVEN_NAME'].split()[1]:
                                b[1] = i['bbox']['left'] + i['bbox']['top']
                    else:
                        if i['value'] == res_pred['GIVEN_NAME'].split()[0]:
                            a[1] = i['bbox']['left'] + i['bbox']['top']
                        elif i['value'] == res_pred['GIVEN_NAME'].split()[1]:
                            b[1] = i['bbox']['left'] + i['bbox']['top']

        return res_pred

    def execute(self, input_payload: Payload, output_payload: Payload,
                context: Context):
        documents = input_payload.get_data()
        for document in documents:
            tagged_documents = []
            name = document["file_name"]
            paragraphs = document["words"]
            for paragraph in paragraphs:
                text = paragraph["value"].split(" ")
                sent = Sentence(text)
                self.model.predict(sent)
                tags = [str(sent[i].labels[0].value) for i in range(len(sent))]
                paragraph["tags"] = tags
            tagged_documents.append(document)

            res_pred = self.tagExtract(tagged_documents)

            res_pred = self.postprocess(document, res_pred)

            tagged_documents = [{'file_name': name, 'result': res_pred}]

            output_payload.add_array(tagged_documents, name=name)
