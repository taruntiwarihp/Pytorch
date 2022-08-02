class Config:

    model_type = "vanilla_transformer_resnet152"
    # Learning Rates
    lr_backbone = 1e-5
    lr = 1e-4

    # Epochs
    epochs = 100
    lr_drop = 20
    start_epoch = 0
    weight_decay = 1e-4

    # Backbone
    backbone = 'resnet152'
    position_embedding = 'sine' # v2 (sine), v3 (learned)
    dilation = True
    
    # Basic
    device = 'cuda'
    seed = 42
    batch_size = 6
    num_workers = 32
    logDir = 'logs'
    modelDir = 'weights'
    checkpoint = 'weights/checkpoint_2.pth'
    clip_max_norm = 0.1

    # Transformer
    hidden_dim = 256 
    pad_token_id = 0
    max_position_embeddings = 128
    activation_fun = "gelu" # "glu" "relu"
    layer_norm_eps = 1e-12
    dropout = 0.1
    vocab_size = 30522 

    enc_layers = 6
    dec_layers = 6
    dim_feedforward = 2048 # resnet fc layer
    nheads = 8 # attentions head
    pre_norm = True

    # Dataset
    dir = 'data/coco'
    limit = -1