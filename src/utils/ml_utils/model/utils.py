from keras import layers, models

def get_embedding_op_dim(inp_dim:int, scale_down_factor:float)->int:
    return int(inp_dim**scale_down_factor) + 1


def Embedding_Layer(vocab_len:int, scale_down_factor:float)->models.Sequential:
    emb_model = models.Sequential()
    emb_model.add(layers.Embedding(input_dim=vocab_len, output_dim=get_embedding_op_dim(vocab_len, scale_down_factor)))
    emb_model.add(layers.Flatten())

    return emb_model