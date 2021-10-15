import re
import inflect
import nltk
import numpy as np
from allennlp.predictors.predictor import Predictor
# from utils.config import args
from nltk.corpus import brown
brown_train = brown.tagged_sents(categories='news')
regexp_tagger = nltk.RegexpTagger(
[(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
(r'(-|:|;)$', ':'),
(r'\'$', 'MD'),
(r'(The|the|A|a|An|an)$', 'AT'),
(r'.able$', 'JJ'),
(r'^[A-Z].$', 'NNP'),
(r'.ness$', 'NN'),
(r'.ly$', 'RB'),
(r'.s$', 'NNS'),
(r'.ing$', 'VBG'),
(r'.ed$', 'VBD'),
(r'.*', 'NN')
])
unigram_tagger = nltk.UnigramTagger(brown_train, backoff=regexp_tagger)
bigram_tagger = nltk.BigramTagger(brown_train, backoff=unigram_tagger)

cfg = {}
cfg["NNP+NNP"] = "NNP"
cfg["NN+NN"] = "NNI"
cfg["NNI+NN"] = "NNI"
cfg["JJ+JJ"] = "JJ"
cfg["JJ+NN"] = "NNI"
class RulePattern():

    def __init__(self):
        super(RulePattern, self).__init__()

        self.pos_useful_tag = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                               'V', 'PDT', 'PRP', 'RBR', 'RBS']
        self.pos_loc_tag = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']
        self.clean = ['The', 'She', 'He', 'They', 'It', 'Them', 'Their', 'A', 'On', 'In', 'To', 'Where', 'There']
        self.no_use_single_words = ['The', 'She', 'He', 'They', 'It', 'Them', 'Their', 'A', 'On', 'In', 'To', 'Where',
                                    'There']
        self.predictor_ner = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz',cuda_device=0)
            # "https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz",cuda_device=0)
        self.predictor_cp = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz',cuda_device=0)
            # "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
        # self.openie_predictor = Predictor.from_path(
        #     "https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
        # self.srl_predictor = Predictor.from_path(
        #     "https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
        self.stemm = nltk.PorterStemmer()
        self.tokenizer = nltk.word_tokenize
        self.inflect = inflect.engine()

    @classmethod
    def check_contain_upper(cls, password):
        pattern = re.compile('[A-Z]+')
        match = pattern.findall(password)
        if match:
            return True
        else:
            return False


    @classmethod
    def get_NP(cls, tree, nps, comparative):

        if isinstance(tree, dict):
            if "children" not in tree:
                if tree['nodeType'] in ["NP", 'JJR', 'JJS','RBR','RBS']:
                    # print(tree['word'])
                    if tree['nodeType'] == 'NP':
                        nps.append(tree['word'])
                    else:
                        comparative.append(tree['word'])
            elif "children" in tree:
                if tree['nodeType'] in ["NP",'JJR','JJS','RBR','RBS']:
                    # print(tree['word'])
                    if tree['nodeType'] == 'NP':
                        nps.append(tree['word'])
                    else:
                        comparative.append(tree['word'])
                    cls.get_NP(tree['children'], nps,comparative)
                else:
                    cls.get_NP(tree['children'], nps,comparative)
        elif isinstance(tree, list):
            for sub_tree in tree:
                cls.get_NP(sub_tree, nps,comparative)

        return nps,comparative

    def normalize_np(self, noun_phrases):
        # noun_phrases = set(noun_phrases)
        cleared_np = []
        for np in noun_phrases:

            if (np not in self.no_use_single_words):
                for item in self.clean:
                    if (np.find(item) == 0):
                        np = np.replace(item + ' ', '')
                        break
                # page = page.replace(' ', '_')

                cleared_np.append(np)
                sin_page = self.inflect.singular_noun(np)
                if (sin_page != False):
                    cleared_np.append(sin_page)

        return cleared_np

    @classmethod
    def get_subjects(cls, tree):
        subject_words = []
        subjects = []
        for subtree in tree['children']:
            if subtree['nodeType'] == "VP" or subtree['nodeType'] == 'S' or subtree['nodeType'] == 'VBZ':
                subjects.append(' '.join(subject_words))
                subject_words.append(subtree['word'])
            else:
                subject_words.append(subtree['word'])
        return subjects

    def found_key_words(self, claims):

        # all_tokens = self.predictor_cp.predict_batch_json(inputs=[{'sentence': text} for text in claims])
        try:
            all_ent_res = self.predictor_ner.predict_batch_json(inputs=[{'sentence': text} for text in claims])
        except Exception as e:
            print(e)
            all_ent_res = []
        try:
            all_tokens = self.predictor_cp.predict_batch_json(inputs=[{'sentence': text} for text in claims])
        except Exception as e:
            print(e)
            all_tokens = []

        all_keywords = []
        for i in range(len(claims)):
            claim = claims[i]
            # tokens = all_tokens[i]
            key_words = {'noun': [], 'content': claim, 'subject': [], 'entity': [],'numbers':[],'comparative':[]}
            if all_ent_res:
                ent_res = all_ent_res[i]
                all_ents = self.extract_entity_allennlp(ent_res['words'], ent_res['tags'])
                key_words['entity'].extend(all_ents)
            # tokens = self.predictior_cp.predict(sentence=claim)
            key_words['numbers'] = re.findall(r"\d+\.?\,?\d*",claim)
            nps,compa = [],[]
            if all_tokens:
                tokens = all_tokens[i]
                tree = tokens['hierplane_tree']['root']
                noun_phrases,comparative = self.get_NP(tree, nps,compa)
                key_words['noun'].extend(noun_phrases)
                key_words['comparative'].extend(comparative)
                subjects = self.get_subjects(tree)
                for subject in subjects:
                    if len(subject) > 0:
                        key_words['subject'].append(subject)

            all_keywords.append(key_words)
        return all_keywords

    @classmethod
    def search_entity_with_tags(cls, tags, words):
        if ('B-V' in tags):
            verb_idx = tags.index('B-V')
        else:
            return [], []
        subj, obj = [], []
        flag = False
        for idx in range(0, verb_idx):
            tag = tags[idx]
            if (tag != 'I-V'):
                if (tag.find('B-') != -1):
                    subj.append(words[idx])
                elif (tag.find('I-') != -1):
                    if (len(subj) != 0):
                        subj[-1] += ' %s' % words[idx]

        for idx in range(verb_idx + 1, len(tags)):
            tag = tags[idx]
            if (tag != 'I-V'):
                if (tag.find('B-') != -1):
                    obj.append(words[idx])
                elif (tag.find('I-') != -1):
                    if (len(obj) != 0):
                        obj[-1] += ' %s' % words[idx]

        return subj, obj
    @classmethod
    def analyze_srl_result(cls, srl_result):
        srls, words = srl_result['verbs'], srl_result['words']
        triples = []
        for srl in srls:
            verb, des, tags = srl['verb'], srl['description'], srl['tags']
            verb = verb
            subj, obj = cls.search_entity_with_tags(tags, words)
            triples.append({'verb': [verb], 'subject': subj, 'object': obj})
        return triples

    def found_openie_srl(self, texts):
        # openie_results = self.openie_predictor.predict_batch_json(inputs=[{'sentence': text} for text in texts])
        srl_results = self.srl_predictor.predict_batch_json(inputs=[{'sentence': text} for text in texts])
        # openie_triples = [self.analyze_srl_result(tmp) for tmp in openie_results]
        # srl_triples = [self.analyze_srl_result(tmp) for tmp in srl_results]
        return  srl_results



    def infer_important_words(self, words, pos_tags):
        # obtain pos tag
        attn_words = []

        def hasNumbers(inputString):
            return bool(re.search(r'\d', inputString))

        for idx in range(len(pos_tags)):
            tag = pos_tags[idx]
            if (tag in self.pos_useful_tag):
                attn_words.append(words[idx])
            elif (hasNumbers(words[idx])):
                attn_words.append(words[idx])

        # if (len(src_loc) > 0 or len(dest_loc) > 0):
        #     print(
        #         'All location candidate is: {}\n The src_loc is: {} The dest_loc is: {}\n'.format(all_loc_cdd, src_loc,
        #                                                                                           dest_loc))

        return attn_words

    def extract_entity_allennlp(self, words, tags):
        all_ents = []
        all_ents_test = []
        flag = True
        # for i, tag in enumerate(tags):
        #     if(tag!='O'):
        #         all_ents_test.append(words[i])
        tmp = []

        for i, tag in enumerate(tags):
            flag = True if (self.check_contain_upper(words[i])) else False
            if (tag != 'O' or flag):
                tmp.append(words[i])
            if (len(tmp) != 0 and ((tag == 'O' and flag == False) or i == (len(tags) - 1))):
                all_ents.append(' '.join(tmp))
                tmp = []

        # assert(' '.join(all_ents_test)==' '.join(all_ents)),'{}, {}, {}, {}'.format(all_ents,all_ents_test,words,tags)
        return all_ents

    @classmethod
    def judge_upper(self, text):
        bigchar = re.findall(r'[A-Z]', text)
        return (len(bigchar) > 0)

    def extract_nltk(self,sentence):
        tokens = self.tokenize_sentence(sentence)
        tags = self.normalize_tags(bigram_tagger.tag(tokens))
        merge = True
        while merge:
            merge = False
            for x in range(0, len(tags) - 1):
                t1 = tags[x]
                t2 = tags[x + 1]
                key = "%s+%s" % (t1[1], t2[1])
                value = cfg.get(key, '')
                # if self.judge_upper(tokens[x]) and self.judge_upper(tokens[x+1]):
                #     value = True
                if value:
                    merge = True
                    tags.pop(x)
                    tags.pop(x)
                    match = "%s %s" % (t1[0], t2[0])
                    pos = value
                    tags.insert(x, (match, pos))
                    break
        matches = []

        for i,t in enumerate(tags):
            if t[1] == "NNP" or t[1] == "NNI":
                # if t[1] == "NNP" or t[1] == "NNI" or t[1] == "NN":

                matches.append(t[0])

        # start = False
        # temp = []
        # for i,t in enumerate(tokens):
        #     if self.judge_upper(t):
        #         if start:
        #             temp.append(t)
        #         else:
        #             start = True
        #             temp.append(t)
        #     else:
        #         if start:
        #             start = False
        #             matches.append(' '.join(temp))
        #             temp = []

        # for i,m1 in enumerate(matches):
        #     for j, m2 in enumerate(matches):
        #         if m1.find(m2)!=-1:
        #             matches.pop(j)


        # for i,match in enumerate(matches):
        #     if (self.judge_upper(match) == False):
        #         matches.pop(i)

        return matches

    def normalize_tags(self, tagged):
        n_tagged = []
        for t in tagged:
            if t[1] == "NP-TL" or t[1] == "NP":
                n_tagged.append((t[0], "NNP"))
                continue
            if t[1].endswith("-TL"):
                n_tagged.append((t[0], t[1][:-3]))
                continue
            if t[1].endswith("S"):
                n_tagged.append((t[0], t[1][:-1]))
                continue
            n_tagged.append((t[0], t[1]))
        return n_tagged

    @classmethod
    def tokenize_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens

    def extract_keywords_nltk(self,sentence):
        kws = self.extract_nltk(sentence)
        key_words = {'keywords': kws, 'sentence': sentence}
        return key_words