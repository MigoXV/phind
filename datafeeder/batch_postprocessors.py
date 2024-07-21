import torch
import torchaudio


class BatchPostProcessor:
    def __init__(self, config):
        n_fft = config.n_fft
        hop_length = config.hop_length
        win_length = config.win_length
        
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length)

    def __call__(self, batch: tuple) -> tuple:
        signals = batch

        spectrograms = [self.postprocess(signal) for signal in signals]
        input_lengths = torch.tensor(
            [spectrogram.shape[0] for spectrogram in spectrograms]
        )

        spectrograms = torch.nn.utils.rnn.pad_sequence(
            spectrograms, batch_first=True, padding_value=0
        )

        return spectrograms, input_lengths

    def postprocess(self, signal: torch.Tensor) -> torch.Tensor:
        spectrogram = self.spectrogram_transform(signal).squeeze(0).T

        # 归一化和标准化
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

        return spectrogram


class TrainBatchPostProcessor(BatchPostProcessor):
    def __call__(self, batch: tuple) -> tuple:
        signals, labels = batch

        spectrograms = [self.postprocess(signal) for signal in signals]
        input_lengths = torch.tensor(
            [spectrogram.shape[0] for spectrogram in spectrograms]
        )

        spectrograms = torch.nn.utils.rnn.pad_sequence(
            spectrograms, batch_first=True, padding_value=0
        )

        return spectrograms, input_lengths, labels