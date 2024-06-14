import pickle
import os
import torch

from collections import defaultdict, Counter
from datasets import load_dataset
from math import log
from tqdm import tqdm


class UnigramLPer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.n_tokens, self.freq_unigram_lp_dict = self.get_freq_unigram_lp_dict()
        self.iso_unigram_lp_dict = self.get_iso_unigram_lp_dict()
        self.aoc_unigram_lp_dict = self.get_aoc_unigram_lp_dict()

        self.iso_additions = 0

    def get_freq_unigram_lp_dict(self):
        if "tok_freqs.pkl" not in os.listdir("unigram_lps"):
            dataset = load_dataset(
                "yhavinga/mc4_nl_cleaned",
                "micro",
                streaming=True,
                trust_remote_code=True,
            )["train"]
            tok_freqs = Counter()
            for doc in tqdm(dataset, total=125000):
                text = doc["text"]
                ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
                tok_freqs.update(ids.tolist())
            pickle.dump(tok_freqs, open("unigram_lps/tok_freqs.pkl", "wb"))

        tok_freqs = pickle.load(open("unigram_lps/tok_freqs.pkl", "rb"))
        n_tokens = sum(tok_freqs.values())
        freq_unigram_lp_dict = {
            tok: log(count / n_tokens) for tok, count in tok_freqs.items()
        }
        return n_tokens, freq_unigram_lp_dict

    def freq(self, tok_id):
        return self.freq_unigram_lp_dict.get(tok_id, log(1 / self.n_tokens))

    def get_iso_unigram_lp_dict(self):
        if "iso_unigram_lp_dict.pkl" in os.listdir("unigram_lps"):
            iso_unigram_lp_dict = pickle.load(
                open("unigram_lps/iso_unigram_lp_dict.pkl", "rb")
            )
        else:
            iso_unigram_lp_dict = {}
        return iso_unigram_lp_dict

    def update_iso_unigram_lp_dict_pickle(self):
        if "iso_unigram_lp_dict.pkl" not in os.listdir("unigram_lps") or len(
            pickle.load(open("unigram_lps/iso_unigram_lp_dict.pkl", "rb"))
        ) < len(self.iso_unigram_lp_dict):
            pickle.dump(
                self.iso_unigram_lp_dict,
                open("unigram_lps/iso_unigram_lp_dict.pkl", "wb"),
            )

    def iso(self, tok_id):
        if tok_id in self.iso_unigram_lp_dict:
            return self.iso_unigram_lp_dict[tok_id]
        tok_str = "<s>" + self.tokenizer.decode(tok_id)
        input_ids = self.tokenizer(tok_str, return_tensors="pt").input_ids[0]
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
        lp = -outputs.loss.item()
        self.iso_unigram_lp_dict[tok_id] = lp

        self.iso_additions += 1
        if self.iso_additions == 100:
            self.update_iso_unigram_lp_dict_pickle()
            self.iso_additions = 0
        return lp

    def get_aoc_unigram_lp_dict(self):
        if "aoc_unigram_lp_dict.pkl" in os.listdir("unigram_lps"):
            aoc_unigram_lp_dict = pickle.load(
                open("unigram_lps/aoc_unigram_lp_dict.pkl", "rb")
            )
        else:
            all_contexts_unigram_lp_dict = defaultdict(list)
            dataset = load_dataset(
                "yhavinga/mc4_nl_cleaned",
                "micro",
                streaming=True,
                trust_remote_code=True,
            )["train"]

            n_docs = 0
            for doc in tqdm(dataset):
                text = doc["text"]
                input_ids = self.tokenizer(text, return_tensors="pt").input_ids[0]

                if len(input_ids) > 1024:
                    continue

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, labels=input_ids)

                for i in range(1, len(input_ids)):
                    tok_lp = torch.log_softmax(outputs.logits[i - 1], dim=-1)[
                        input_ids[i]
                    ].item()
                    all_contexts_unigram_lp_dict[input_ids[i].item()].append(tok_lp)

                n_docs += 1

                if n_docs == 23000:
                    aoc_unigram_lp_dict = {
                        tok_id: sum(tok_lps) / len(tok_lps)
                        for tok_id, tok_lps in all_contexts_unigram_lp_dict.items()
                    }
                    pickle.dump(
                        all_contexts_unigram_lp_dict,
                        open(f"unigram_lps/all_contexts_unigram_lp_dict.pkl", "wb"),
                    )
                    pickle.dump(
                        aoc_unigram_lp_dict,
                        open(f"unigram_lps/aoc_unigram_lp_dict.pkl", "wb"),
                    )
                    break
        return aoc_unigram_lp_dict

    def aoc(self, tok_id):
        return self.aoc_unigram_lp_dict.get(tok_id, self.iso(tok_id))
