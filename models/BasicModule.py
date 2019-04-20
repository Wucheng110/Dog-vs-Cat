import time
import torch as t
from torch import nn

class BasicModule(nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))
        print(self.model_name)

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        '''
        保存模型，默认用'模型名 + 时间' 命名
        '''
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name

# if __name__ == '__main__':
#     a = BasicModule()