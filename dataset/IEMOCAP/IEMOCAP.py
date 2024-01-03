import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Config, Wav2Vec2Model

config = Wav2Vec2Config()
model = Wav2Vec2Model(config)


def Partition(ith_fold, DataMap):
    '''
    Args:
        ith_fold: the test session for iemocap, ranging from 1 to 5
    '''
    train_list = []
    test_list = DataMap['Session{}'.format(ith_fold)]
    for i in range(1, 6):
        sess_name = 'Session{}'.format(i)
        if i != ith_fold:
            train_list.extend(DataMap[sess_name])
    train_list = list(filter(lambda x: x['audio_length'] < 300000, train_list)) # For training efficiency
    train_dataset = IEMOCAP(train_list)
    test_dataset = IEMOCAP(test_list)
    return train_dataset, test_dataset

def Merge(DataMap):
    train_list = []
    for i in range(1, 6):
        sess_name = 'Session{}'.format(i)
        train_list.extend(DataMap[sess_name])
    train_list = list(filter(lambda x: x['audio_length'] < 300000, train_list)) # For training efficiency
    train_dataset = IEMOCAP(train_list)
    return train_dataset

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


class IEMOCAP(data.Dataset):
    """Speech dataset."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['audio'], self.data[index]['audio_length'], self.data[index]['tlabel'], self.data[index][
            'label'], self.data[index]['text']

    def collate_fn(self, datas):
        ''' Padding audio and tlabel dynamically, computing length of pretrained features
        Args:
            datas: List[(index: int, audio: torch.Tensor, audio_length: int, tlabel: List[int], label: int, plabel: torch.Tensor, text: str)]
        '''
        audio = [data[0] for data in datas]
        padded_audio = pad_sequence(audio, batch_first=True, padding_value=0)
        audio_length = torch.tensor([get_feat_extract_output_lengths(data[1]) for data in datas])
        tlabel = [torch.tensor(data[2]) for data in datas]
        tlabel_length = torch.tensor([tokens.size(0) for tokens in tlabel])
        padded_tlabel = pad_sequence(tlabel, batch_first=True, padding_value=-1)
        label = torch.tensor([data[3] for data in datas])
        text = [data[4] for data in datas]

        return padded_audio, audio_length, padded_tlabel, tlabel_length, label, text
