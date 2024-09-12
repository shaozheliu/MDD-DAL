import torch
import torch.nn as nn
from models.MyTransformer import ST_Transformer_feature_extractor,CNN_Embedding, CNN_Embedding_Deepconv, \
    ST_Transformer_feature_extractor_sec, CNN_Embedding_Deepconv_2b, ST_Transformer_feature_extractor_2b
from torch.autograd import Function



class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None





class Mymodel(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout, emb_feature, type = 'all', clf_class=2, domain_class=2):
        super(Mymodel, self).__init__()

        ## Embedding layer ##
        self.embedding = nn.Sequential()
        self.embedding.add_module('CNN_embedding',CNN_Embedding_Deepconv_2b(19, emb_feature, dropout))

        ## Feature extraction layer
        self.feature = nn.Sequential()
        self.feature.add_module('Transformer extractor',ST_Transformer_feature_extractor_2b(d_model, d_ff, n_heads, n_layers, dropout,emb_feature, type))

        ## method2
        # Class classifier layer
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(915, clf_class))

        # Domain classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(915, domain_class))


    def forward(self, x, alpha):
        x = self.embedding(x)  # （2，20，244）
        feature = self.feature(x)
        # feature = x + feature   # 残差
        # feature = feature.view(-1,feature.shape[1] * feature.shape[2])  # 将维度拉平
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output





if __name__ == "__main__":
    inp = torch.autograd.Variable(torch.randn(2, 19, 1024))
    head = 4
    d_model = 244
    d_ff = 512
    emb_feature = 20
    type = 'all'
    ff_hide = 1024
    mode1 = "T"
    mode2 = "C"
    n_layer = 3
    dropout = 0.1
    alpha = 0.02
    model = Mymodel(d_model, d_ff, head, n_layer, dropout, emb_feature, type)
    class_output, domain_output  = model(inp, alpha)