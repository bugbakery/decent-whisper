# Vendored from https://github.com/ml-explore/mlx-examples/blob/6120a5f3763788f2444e082875b917925b80afa5/whisper/mlx_whisper/transcribe.py to add support for beam search
# Copyright © 2023 Apple Inc.

from dataclasses import replace
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from mlx_whisper.audio import CHUNK_LENGTH
from mlx_whisper.decoding import (
    ApplyTimestampRules,
    DecodingOptions,
    DecodingResult,
    GreedyDecoder,
    Inference,
    LogitFilter,
    MaximumLikelihoodRanker,
    SequenceRanker,
    SuppressBlank,
    SuppressTokens,
    TokenDecoder,
    compression_ratio,
)
from mlx_whisper.tokenizer import Tokenizer, get_tokenizer
from mlx_whisper.whisper import Whisper


# stolen from https://github.com/altalt-org/lightning-whisper-mlx-beamsearch/blob/5d4d906/decoding.py#L286
class BeamSearchDecoder(TokenDecoder):
    def __init__(
        self,
        beam_size: int,
        eot: int,
        inference: Inference,
        patience: Optional[float] = None,
    ):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None

        assert self.max_candidates > 0, (
            f"Invalid patience value: {self.patience}, results in {self.max_candidates} candidates"
        )

    def reset(self):
        self.finished_sequences = None

    def update(
        self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
    ) -> Tuple[mx.array, bool, mx.array]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:
            self.finished_sequences = [{} for _ in range(n_audio)]

        # log-softmax
        logprobs = logits.astype(mx.float32) - mx.logsumexp(
            logits.astype(mx.float32), axis=-1, keepdims=True
        )

        k = self.beam_size + 1
        sorted_indices = mx.argsort(logprobs, axis=-1)
        top_tokens = sorted_indices[..., -k:]
        top_tokens = top_tokens[..., ::-1]
        top_logprobs = mx.take_along_axis(logprobs, top_tokens, -1)

        scores = sum_logprobs[:, None] + top_logprobs
        scores = scores.reshape(n_audio, self.beam_size * k)
        top_tokens = top_tokens.reshape(n_audio, self.beam_size * k)

        source_beams_per_audio = mx.arange(self.beam_size)[:, None]
        source_beams_per_audio = mx.broadcast_to(
            source_beams_per_audio, (self.beam_size, k)
        ).flatten()

        sorted_indices = mx.argsort(scores, axis=-1)

        selected_sources = []
        selected_tokens = []
        selected_scores = []

        # Track newly finished sequences per audio
        newly_finished_all = [{} for _ in range(n_audio)]

        for i in range(n_audio):
            picks = 0
            # iterate from highest to lowest score
            for idx in reversed(sorted_indices[i].tolist()):
                src = source_beams_per_audio[idx].item()
                tok = top_tokens[i, idx].item()
                global_src = i * self.beam_size + src
                seq = tokens[global_src].tolist() + [tok]
                sc = float(scores[i, idx].item())

                if tok == self.eot:
                    # record finished; do not keep in active set
                    if tuple(seq) not in self.finished_sequences[i]:
                        newly_finished_all[i][tuple(seq)] = sc
                    continue

                selected_sources.append(global_src)
                selected_tokens.append(tok)
                selected_scores.append(sc)
                picks += 1
                if picks == self.beam_size:
                    break

        # Reorder KV cache according to selected sources
        self.inference.rearrange_kv_cache(selected_sources)

        # Build next tokens tensor and sum_logprobs
        base = tokens[selected_sources]
        tokens = mx.concatenate([base, mx.array(selected_tokens)[:, None]], axis=1)
        sum_logprobs = mx.array(selected_scores, dtype=mx.float32)

        # Merge newly finished sequences into the pool with patience
        for i in range(n_audio):
            previously_finished = self.finished_sequences[i]
            newly_finished = newly_finished_all[i]
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break
                previously_finished.setdefault(seq, newly_finished[seq])

        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        return tokens, completed, sum_logprobs

    def finalize(self, tokens: mx.array, sum_logprobs: mx.array):
        n_audio = tokens.shape[0]
        beam_size = tokens.shape[1]

        # if not enough finished sequences, append EOT to top unfinished candidates
        for i in range(n_audio):
            if len(self.finished_sequences[i]) < beam_size:
                order = mx.argsort(sum_logprobs[i]).tolist()[::-1]
                for j in order:
                    seq = tokens[i, int(j)].tolist() + [self.eot]
                    score = float(sum_logprobs[i, int(j)].item())
                    tseq = tuple(seq)
                    if tseq not in self.finished_sequences[i]:
                        self.finished_sequences[i][tseq] = score
                    if len(self.finished_sequences[i]) >= beam_size:
                        break

        tokens_out = [
            [list(seq) for seq in sequences.keys()]
            for sequences in self.finished_sequences
        ]
        sum_out = [
            [float(score) for score in sequences.values()]
            for sequences in self.finished_sequences
        ]
        return tokens_out, sum_out


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task=options.task,
        )
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = Inference(model)

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = self.decoder = BeamSearchDecoder(
                options.beam_size, tokenizer.eot, self.inference, options.patience
            )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(
                SuppressBlank(self.tokenizer, self.sample_begin, model.dims.n_vocab)
            )
        if self.options.suppress_tokens:
            self.logit_filters.append(
                SuppressTokens(self._get_suppress_tokens(), model.dims.n_vocab)
            )

        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: mx.array):
        if self.options.fp16:
            mel = mel.astype(mx.float16)

        if mel.shape[-2:] == (
            self.model.dims.n_audio_ctx,
            self.model.dims.n_audio_state,
        ):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            audio_features = self.model.encoder(mel)

        if audio_features.dtype != (mx.float16 if self.options.fp16 else mx.float32):
            raise TypeError(
                f"audio_features has an incorrect dtype: {audio_features.dtype}"
            )

        return audio_features

    def _detect_language(self, audio_features: mx.array, tokens: np.array):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(
                audio_features, self.tokenizer
            )
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                # write language tokens
                tokens[:, self.sot_index + 1] = np.array(lang_tokens)

        return languages, lang_probs

    def _main_loop(self, audio_features: mx.array, tokens: mx.array):
        n_batch = tokens.shape[0]
        sum_logprobs = mx.zeros(n_batch)

        def _step(inputs, audio_features, tokens, sum_logprobs):
            pre_logits = self.inference.logits(inputs, audio_features)

            # consider the logits at the last token only
            logits = pre_logits[:, -1]

            # apply the logit filters, e.g. for suppressing or applying penalty to
            for logit_filter in self.logit_filters:
                logits = logit_filter.apply(logits, tokens)

            # expand the tokens tensor with the selected next tokens
            tokens, completed, sum_logprobs = self.decoder.update(
                tokens, logits, sum_logprobs
            )
            return tokens, completed, sum_logprobs, pre_logits

        tokens, completed, sum_logprobs, pre_logits = _step(
            tokens, audio_features, tokens, sum_logprobs
        )
        if self.tokenizer.no_speech is not None:  # compute no_speech_probs
            probs_at_sot = mx.softmax(pre_logits[:, self.sot_index], axis=-1)
            no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech]
        else:
            no_speech_probs = mx.full(n_batch, mx.nan)
        mx.async_eval(completed, tokens, sum_logprobs, no_speech_probs)

        for i in range(1, self.sample_len):
            inputs = tokens[:, -1:]
            if tokens.shape[-1] > self.n_ctx:
                break

            next_tokens, next_completed, next_sum_logprobs, _ = _step(
                inputs, audio_features, tokens, sum_logprobs
            )
            mx.async_eval(next_completed, next_tokens, next_sum_logprobs)
            if completed:
                break
            tokens = next_tokens
            completed = next_completed
            sum_logprobs = next_sum_logprobs

        return tokens, sum_logprobs, no_speech_probs

    def run(self, mel: mx.array) -> List[DecodingResult]:
        self.inference.reset()
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features: mx.array = self._get_audio_features(mel)  # encoder forward pass
        tokens: mx.array = mx.array(self.initial_tokens)
        tokens = mx.broadcast_to(tokens, (n_audio, len(self.initial_tokens)))

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]

        # repeat tokens by the group size, for beam search or best-of-n sampling
        if self.n_group > 1:
            tokens = tokens[:, None, :]
            tokens = mx.broadcast_to(
                tokens, [n_audio, self.n_group, len(self.initial_tokens)]
            )
            tokens = tokens.reshape((n_audio * self.n_group, len(self.initial_tokens)))

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)

        if isinstance(self.decoder, GreedyDecoder):
            tokens = tokens[..., self.sample_begin :]
            # eval and convert to list
            mx.eval(tokens, sum_logprobs, no_speech_probs)
            tokens = tokens.tolist()
            sum_logprobs = sum_logprobs.tolist()
            no_speech_probs = no_speech_probs.tolist()
            tokens = [[t[: t.index(tokenizer.eot)] for t in s] for s in tokens]
        else:
            tokens = [[t[self.sample_begin :] for t in s] for s in tokens]
            tokens = [
                [t[: t.index(tokenizer.eot) if tokenizer.eot in t else len(t)] for t in s]
                for s in tokens
            ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i] for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


def decode(
    model: "Whisper",
    mel: mx.array,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: mx.array, shape = (80, 3000) or (*, 80, 3000)
        An array containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    if single := mel.ndim == 2:
        mel = mel[None]

    if kwargs:
        options = replace(options, **kwargs)

    result = DecodingTask(model, options).run(mel)
    return result[0] if single else result
