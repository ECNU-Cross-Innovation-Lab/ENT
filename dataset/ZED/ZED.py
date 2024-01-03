import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Config, Wav2Vec2Model

config = Wav2Vec2Config()
model = Wav2Vec2Model(config)


def get_feat_extract_output_lengths(input_length):
    """
        Computes the output length of the convolutional layers
        """
    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return (input_length - kernel_size) // stride + 1

    for kernel_size, stride in zip(model.config.conv_kernel, model.config.conv_stride):
        input_length = _conv_out_length(input_length, kernel_size, stride)
    return input_length


class ZED(data.Dataset):
    """Speech dataset."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['id'], self.data[index]['audio'], self.data[index]['audio_length'], self.data[index]['duration'], self.data[
            index]['frame_label'], self.data[index]['interval'], self.data[index]['emotion'], self.data[index]['text']

    def collate_fn(self, datas):
        ''' Padding audio dynamically, computing length of pretrained features
        Args:
            datas: List[(id: str, audio: torch.Tensor, audio_length: int, duration: float, frame_label: List[str], interval: Dict {'emo':str, 'start':float, 'end': float}, emotion: int, text: str)]
        '''
        id = [data[0] for data in datas]
        audio = [data[1] for data in datas]
        padded_audio = pad_sequence(audio, batch_first=True, padding_value=0)
        audio_length = torch.tensor([get_feat_extract_output_lengths(data[2]) for data in datas])
        duration = [data[3] for data in datas]
        frame_label = [data[4] for data in datas]
        interval = [data[5] for data in datas]
        emotion = torch.tensor([data[6] for data in datas])
        text = [data[7] for data in datas]

        return id, padded_audio, audio_length, duration, frame_label, interval, emotion, text
