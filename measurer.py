from unigram_lper import UnigramLPer

import torch


class Measurer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.unigram_lper = UnigramLPer(model, tokenizer)

    def get_measures(self, sent, print_info=False):
        input_ids = self.tokenizer(sent, return_tensors="pt").input_ids[0]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)

        tok_lps = [
            torch.log_softmax(outputs.logits[i - 1], dim=-1)[input_ids[i]].item()
            for i in range(1, len(input_ids))
        ]

        sent_len = len(tok_lps)
        lp = sum(tok_lps)

        unigram_lps_freq = [self.unigram_lper.freq(id) for id in input_ids[1:].tolist()]
        unigram_lps_iso = [self.unigram_lper.iso(id) for id in input_ids[1:].tolist()]
        unigram_lps_aoc = [self.unigram_lper.aoc(id) for id in input_ids[1:].tolist()]

        results = [sent_len]

        for unigram_lps in [unigram_lps_freq, unigram_lps_iso, unigram_lps_aoc]:
            unigram_lp = sum(unigram_lps)

            norm_tok_lps = sorted(
                -(tok_lp / unigram_lp)
                for tok_lp, unigram_lp in zip(tok_lps, unigram_lps)
            )

            q1 = sent_len // 4 if sent_len >= 4 else 1
            q2 = sent_len // 2 if sent_len >= 2 else 1

            # Sentence-level measures
            mean_lp = lp / sent_len
            norm_lp_div = -(lp / unigram_lp)
            norm_lp_sub = lp - unigram_lp
            slor = norm_lp_sub / sent_len

            # Word-level measures
            (
                word_lp_min_1,
                word_lp_min_2,
                word_lp_min_3,
                word_lp_min_4,
                word_lp_min_5,
            ) = (norm_tok_lps + norm_tok_lps[-1:] * 5)[:5]
            word_lp_mean = sum(norm_tok_lps) / sent_len
            word_lp_mean_q1 = sum(norm_tok_lps[:q1]) / q1
            word_lp_mean_q2 = sum(norm_tok_lps[:q2]) / q2
            word_lp_mean_sq = (
                -sum(norm_tok_lp**2 for norm_tok_lp in norm_tok_lps) / sent_len
            )

            results.extend(
                [
                    unigram_lp,
                    lp,
                    mean_lp,
                    norm_lp_div,
                    norm_lp_sub,
                    slor,
                    word_lp_min_1,
                    word_lp_min_2,
                    word_lp_min_3,
                    word_lp_min_4,
                    word_lp_min_5,
                    word_lp_mean,
                    word_lp_mean_q1,
                    word_lp_mean_q2,
                    word_lp_mean_sq,
                ]
            )

        if print_info:
            self.print_info(
                sent_len,
                input_ids,
                tok_lps,
                unigram_lps_freq,
                unigram_lps_iso,
                unigram_lps_aoc,
            )

        return results

    def print_info(
        self,
        sent_len,
        input_ids,
        tok_lps,
        unigram_lps_freq,
        unigram_lps_iso,
        unigram_lps_aoc,
    ):
        print("Sentence length:", sent_len)
        print()
        print(
            f"{'':<15} {'LP':<10} {'Unigram LP (freq)':<18} {'Unigram LP (iso)':<18} {'Unigram LP (aoc)':<18}"
        )
        for i in range(sent_len):
            print(
                f"{self.tokenizer.decode(input_ids[i+1]):<15}",
                f"{tok_lps[i]:<10.4f}",
                f"{unigram_lps_freq[i]:<18.4f}",
                f"{unigram_lps_iso[i]:<18.4f}",
                f"{unigram_lps_aoc[i]:<18.4f}",
            )
        print(
            f"{'Total':<15}",
            f"{sum(tok_lps):<10.4f}",
            f"{sum(unigram_lps_freq):<18.4f}",
            f"{sum(unigram_lps_iso):<18.4f}",
            f"{sum(unigram_lps_aoc):<18.4f}",
        )
